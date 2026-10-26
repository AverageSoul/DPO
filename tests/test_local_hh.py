"""Local HH loading must work without parent-process monkeypatches."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import preference_datasets as data


class LocalHHTests(unittest.TestCase):
    def test_default_hub(self):
        with patch.object(data.datasets, 'load_dataset') as load:
            data.load_hh_split('train')
            load.assert_called_once_with('Anthropic/hh-rlhf', split='train', cache_dir=None)

    def test_raw_nested_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            category = root / 'helpful-base'
            category.mkdir()
            raw = category / 'train.json.gz'
            raw.touch()
            with patch.object(data.datasets, 'load_dataset') as load:
                data.load_hh_split('train', dataset_path=directory)
                load.assert_called_once_with('json', data_files={'train': [str(raw)]},
                                             split='train', cache_dir=None)

    def test_saved_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'dataset_dict.json'
            marker.touch()
            with patch.object(data.datasets, 'load_from_disk', return_value={'test': 'saved'}) as load:
                self.assertEqual(data.load_hh_split('test', dataset_path=directory), 'saved')
                load.assert_called_once_with(str(root))
            marker.unlink()
            (root / 'test').mkdir()
            (root / 'test' / 'state.json').touch()
            with patch.object(data.datasets, 'load_from_disk', return_value='split') as load:
                self.assertEqual(data.load_hh_split('test', dataset_path=directory), 'split')
                load.assert_called_once_with(str(root / 'test'))

    def test_reject_single_saved_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'state.json').touch()
            with self.assertRaisesRegex(ValueError, 'both train and test'):
                data.load_hh_split('test', dataset_path=directory)

    def test_dispatch_propagates_path(self):
        record = {'prompt': {'responses': ['a', 'b'], 'pairs': [(0, 1)], 'sft_target': 'a'}}
        with patch.object(data, 'get_hh', return_value=record) as load:
            data.get_dataset('hh', 'train', dataset_path='/local/hh')
            load.assert_called_once_with('train', silent=False, cache_dir=None, dataset_path='/local/hh')
        with self.assertRaisesRegex(ValueError, 'only for hh'):
            data.get_dataset('shp', 'train', dataset_path='/local/hh')


if __name__ == '__main__':
    unittest.main()
