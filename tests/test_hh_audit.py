import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import torch
from hh_split_audit import audit_hh_splits, split_hh_row
from preference_datasets import get_batch_iterator
from trainers import BasicTrainer
from omegaconf import OmegaConf


def hh(prompt, chosen=' good', rejected=' bad'):
    prefix = f'\n\nHuman: {prompt}\n\nAssistant:'
    return dict(chosen=prefix + chosen, rejected=prefix + rejected)


def distributed_accuracy_worker(rank, init_file, output):
    import torch.distributed as dist
    from datetime import timedelta
    os.environ['USE_LIBUV'] = '0'
    dist.init_process_group('gloo', init_method=Path(init_file).as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        trainer = BasicTrainer.__new__(BasicTrainer)
        trainer.rank, trainer.world_size = rank, 2
        trainer.policy, trainer.reference_model = object(), object()
        cfg = OmegaConf.create(dict(name='dpo', beta=0.1, reference_free=False,
                                    label_smoothing=0., mode_loss='mean'))
        local = torch.tensor([1., -1.]) if rank == 0 else torch.tensor([-2., -3.])
        trainer.concatenated_forward = Mock(side_effect=[(local, torch.zeros(2)),
                                                       (torch.zeros(2), torch.zeros(2))])
        _, metrics = trainer.get_batch_metrics({}, cfg, train=False)
        summary = trainer.summarize_metrics(metrics)
        # Both ranks receive all four pairs; rank 0 writes one global result.
        assert metrics['preference_acc/eval'] == [1., 0., 0., 0.]
        assert summary['preference_acc/eval'] == 0.25
        assert len(metrics['logps_eval/chosen']) == 4
        if rank == 0:
            Path(output).write_text(json.dumps(summary), encoding='utf-8')
    finally:
        dist.destroy_process_group()


class HHAuditTests(unittest.TestCase):
    def test_response_continuation_does_not_change_shared_prompt(self):
        row = hh('question', ' answer\n\nHuman: followup\n\nAssistant: next', ' other')
        prompt, chosen, rejected = split_hh_row(row)
        self.assertEqual(prompt, '\n\nHuman: question\n\nAssistant:')
        self.assertTrue(chosen.endswith('Assistant: next'))
        self.assertEqual(rejected, ' other')
        splits, report = audit_hh_splits([hh('question')], [row])
        self.assertEqual(splits['train'], [])
        self.assertEqual(report['train_pairs_removed'], 1)

    def test_heldout_prompts_and_unique_unordered_pairs(self):
        test = [hh('held out'), hh('held out', ' bad', ' good'),
                hh('held out', ' second good', ' second bad')]
        train = [hh('held out'), hh('held   out', ' another', ' other'), hh('train only')]
        splits, report = audit_hh_splits(train, test)
        self.assertEqual(splits['train'], [train[2]])
        self.assertEqual(splits['test'], [test[0], test[2]])
        self.assertEqual(report['test_raw_pairs'], 3)
        self.assertEqual(report['test_unique_pairs'], 2)
        self.assertEqual(report['train_pairs_removed'], 2)
        self.assertEqual(report['remaining_prompt_overlap'], 0)

    def test_1000_pairs_do_not_take_train_samples(self):
        test = [hh(f'test {i}') for i in range(1003)]
        train = [hh('train only'), hh('test 10')]
        splits, report = audit_hh_splits(train, test)
        self.assertEqual(report['test_unique_pairs'], 1003)
        self.assertEqual(len(splits['train']), 1)
        data = {str(i): dict(responses=['a', 'b'], pairs=[(0, 1)], sft_target='a')
                for i in range(len(splits['test']))}
        def tokenize(prompt, *args):
            return {'chosen_input_ids': [int(prompt)]}
        with patch('preference_datasets.get_dataset', return_value=data) as load, patch(
                'preference_datasets.tokenize_batch_element', side_effect=tokenize):
            batches = list(get_batch_iterator(['hh'], NS(pad_token_id=0), split='test',
                batch_size=4, n_epochs=1, n_examples=1000, seed=0, silent=True))
        seen = [int(x) for batch in batches for x in batch['chosen_input_ids'].flatten()]
        self.assertEqual(len(batches), 250)
        self.assertEqual(len(seen), 1000)
        self.assertEqual(len(set(seen)), 1000)
        self.assertEqual(load.call_args.args[1], 'test')

    def test_two_rank_global_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            init_file = str(Path(directory) / 'init')
            output = str(Path(directory) / 'global.json')
            torch.multiprocessing.spawn(distributed_accuracy_worker,
                                        args=(init_file, output), nprocs=2, join=True)
            metrics = json.loads(Path(output).read_text(encoding='utf-8'))
            self.assertEqual(metrics['preference_acc/eval'], 0.25)
            self.assertEqual(metrics['policy_preference_acc/eval'], 0.25)


if __name__ == '__main__':
    unittest.main()
