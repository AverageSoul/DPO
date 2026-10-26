import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import torch
from omegaconf import OmegaConf
from trainers import BasicTrainer


class PreferenceEvalTests(unittest.TestCase):
    def test_accuracy_definitions_and_reference_ties(self):
        trainer = BasicTrainer.__new__(BasicTrainer)
        trainer.rank = 0
        trainer.world_size = 1
        trainer.policy = object()
        trainer.reference_model = object()
        cfg = OmegaConf.create(dict(name='dpo', beta=0.1, reference_free=False,
                                    label_smoothing=0.0, mode_loss='mean'))
        # Relative margins: [1, -1, 0]; raw policy margins: [-1, 1, 0].
        pc, pr = torch.tensor([-3., -2., -4.]), torch.tensor([-2., -3., -4.])
        rc, rr = torch.tensor([-4., -1., -4.]), torch.tensor([-2., -3., -4.])
        trainer.concatenated_forward = Mock(side_effect=[(pc, pr), (rc, rr)])
        _, metrics = trainer.get_batch_metrics({}, cfg, train=False)
        self.assertEqual(metrics['preference_acc/eval'], [1., 0., 0.])
        self.assertEqual(metrics['policy_preference_acc/eval'], [0., 1., 0.])
        self.assertEqual(metrics['preference_acc/eval'], metrics['rewards_eval/accuracies'])
        trainer.concatenated_forward = Mock(side_effect=[(pc, pr), (pc, pr)])
        _, metrics = trainer.get_batch_metrics({}, cfg, train=False)
        self.assertEqual(metrics['preference_acc/eval'], [0., 0., 0.])

    def test_final_only_save_and_eval_count(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = OmegaConf.create(dict(
                trainer='BasicTrainer', seed=0, datasets=['toy'], noise_rate=0,
                batch_size=2, eval_batch_size=2, max_length=6, max_prompt_length=2,
                n_epochs=1, n_examples=4, n_eval_examples=2, optimizer='RMSprop',
                lr=0.001, warmup_steps=0, max_grad_norm=1., eval_every=2,
                gradient_accumulation_steps=1, sample_during_eval=False,
                do_first_eval=True, minimum_log_interval_secs=1e9, debug=False,
                wandb={'enabled': False}, local_dirs=[directory],
                model={'name_or_path': 'toy', 'tokenizer_name_or_path': None},
                loss=dict(name='dpo', beta=0.1, label_smoothing=0., reference_free=False, mode_loss='mean')))
            batch = {'chosen_input_ids': torch.ones(2, 3, dtype=torch.long)}
            def iterator(**kwargs):
                if kwargs['split'] == 'test':
                    self.assertEqual(kwargs['n_epochs'], 1)
                    return iter([batch])
                return iter([batch, batch])
            model = torch.nn.Linear(1, 1).to('cuda' if torch.cuda.is_available() else 'cpu')
            with patch('trainers.get_batch_iterator', side_effect=iterator), patch(
                    'trainers.transformers.AutoTokenizer.from_pretrained', return_value=NS(pad_token_id=0)):
                trainer = BasicTrainer(model, cfg, 0, directory, torch.nn.Linear(1, 1))
            def forward(policy, batch, **kwargs):
                if policy is trainer.policy:
                    p = policy.weight.reshape(1).expand(2)
                    return p, p - 1
                return torch.zeros(2, device=trainer.device), torch.zeros(2, device=trainer.device)
            trainer.concatenated_forward = forward
            with patch.object(trainer, 'save', wraps=trainer.save) as save:
                trainer.train()
                save.assert_not_called()
                trainer.save(metrics=trainer.last_eval_metrics)
                save.assert_called_once()
            records = [json.loads(line) for line in (Path(directory) / 'metrics.jsonl').read_text().splitlines()]
            evaluations = [r for r in records if r['phase'] == 'eval']
            self.assertEqual([r['examples'] for r in evaluations], [0, 2, 4])
            self.assertEqual([r['metrics']['eval/examples'] for r in evaluations], [2, 2, 2])
            self.assertTrue(all(r['metrics']['preference_acc/eval'] == 1. for r in evaluations))
            self.assertEqual(list(Path(directory).glob('step-*')), [])
            state = torch.load(Path(directory) / 'LATEST' / 'policy.pt', map_location='cpu', weights_only=False)
            self.assertEqual(state['metrics'], trainer.last_eval_metrics)
            self.assertEqual(state['step_idx'], 4)


if __name__ == '__main__':
    unittest.main()
