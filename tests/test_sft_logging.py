import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from transformers import GPT2Config, GPT2LMHeadModel
from trainers import BasicTrainer


class SFTLoggingTests(unittest.TestCase):
    def test_token_weighted_metrics(self):
        metrics = BasicTrainer.summarize_metrics({'loss/eval': [4.0, 12.0],
                                                  'tokens_eval/response': [2, 3]})
        self.assertAlmostEqual(metrics['nll/eval'], 16 / 5)
        self.assertAlmostEqual(metrics['perplexity/eval'], math.exp(16 / 5))

    def test_fsdp_sft_save_skips_optimizer_collection(self):
        import contextlib
        from unittest.mock import Mock
        from trainers import FSDPTrainer
        trainer = FSDPTrainer.__new__(FSDPTrainer)
        trainer.config = OmegaConf.create({'loss': {'name': 'sft'}})
        trainer.rank = 0
        trainer.example_counter = 4
        trainer.policy = Mock()
        trainer.policy.state_dict.return_value = {'weight': torch.ones(2)}
        trainer.write_state_dict = Mock()
        # No optimizer exists: the SFT save path must never try to collect its state.
        with patch('trainers.FSDP.state_dict_type', return_value=contextlib.nullcontext()), patch(
                'trainers.dist.barrier') as barrier, patch('trainers.FSDP.optim_state_dict') as optim:
            trainer.save(metrics={'nll/eval': 2.0})
            trainer.write_state_dict.assert_called_once_with(
                4, {'weight': trainer.policy.state_dict.return_value['weight']},
                {'nll/eval': 2.0}, 'policy.pt', None)
            barrier.assert_called_once()
            optim.assert_not_called()

    def test_actual_loop_logs_all_updates_and_final_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = OmegaConf.create(dict(
                trainer='BasicTrainer', seed=0, datasets=['toy'], noise_rate=0,
                batch_size=2, eval_batch_size=2, max_length=6, max_prompt_length=2,
                n_epochs=1, n_examples=4, n_eval_examples=2, optimizer='RMSprop',
                lr=0.001, warmup_steps=0, max_grad_norm=1.0, eval_every=2,
                gradient_accumulation_steps=1, sample_during_eval=False,
                do_first_eval=True, minimum_log_interval_secs=1e9, debug=False,
                wandb={'enabled': False}, local_dirs=[directory],
                model={'name_or_path': 'toy', 'tokenizer_name_or_path': None},
                loss={'name': 'sft', 'mode_loss': 'mean'}))
            torch.manual_seed(4)
            model = GPT2LMHeadModel(GPT2Config(vocab_size=19, n_embd=8,
                                              n_layer=1, n_head=2, n_positions=16))
            model.to('cuda' if torch.cuda.is_available() else 'cpu')
            ids = torch.randint(0, 19, (2, 6))
            labels = ids.clone(); labels[:, :2] = -100
            batch = {'chosen_input_ids': ids, 'chosen_labels': labels,
                     'chosen_attention_mask': torch.ones_like(ids), 'prompt': ['a', 'b']}
            def iterator(**kwargs):
                return iter([batch, batch] if kwargs['split'] == 'train' else [batch])
            with patch('trainers.get_batch_iterator', side_effect=iterator), patch(
                    'trainers.transformers.AutoTokenizer.from_pretrained', return_value=NS(pad_token_id=0)):
                trainer = BasicTrainer(model, cfg, 0, directory)
                with patch.object(trainer, 'write_state_dict', wraps=trainer.write_state_dict) as write:
                    trainer.train()
                    write.assert_not_called()
                    trainer.save(metrics=trainer.last_eval_metrics)
                    write.assert_called_once()
                records = [json.loads(line) for line in (Path(directory) / 'metrics.jsonl').read_text().splitlines()]
                self.assertEqual([r['examples'] for r in records if r['phase'] == 'train'], [2, 4])
                self.assertEqual([r['examples'] for r in records if r['phase'] == 'eval'], [0, 2, 4])
                self.assertEqual(list(Path(directory).glob('step-*')), [])
                self.assertEqual(sorted(p.name for p in (Path(directory) / 'LATEST').iterdir()), ['policy.pt'])
                state = torch.load(Path(directory) / 'LATEST' / 'policy.pt', map_location='cpu', weights_only=False)
                self.assertEqual(state['step_idx'], 4)
                self.assertEqual(state['metrics'], trainer.last_eval_metrics)
                # Optional samples persist locally even with W&B disabled.
                cfg.sample_during_eval = True
                cfg.n_eval_model_samples = 2
                with patch.object(trainer, 'get_batch_samples', return_value=(['answer a', 'answer b'], [])):
                    trainer.evaluate()
                samples = json.loads((Path(directory) / 'samples' / 'step-4.json').read_text())
                self.assertEqual(samples['prompts'], ['a', 'b'])
                self.assertEqual(samples['policy_samples'], ['answer a', 'answer b'])


if __name__ == '__main__':
    unittest.main()
