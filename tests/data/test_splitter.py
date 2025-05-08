import importlib
from pathlib import Path
from tests.utils import *

import pytest

current_path = Path(__file__).resolve().parent


class TestSplitter:

    _dataset_path = str(current_path / 'splitting_strategies/dataset.tsv')

    def _apply_splitter(self, config):
        ns = create_namespace(config)
        if not hasattr(self, '_sample_df'):
            cols = ['userId', 'timestamp']
            self._sample_df = read_dataset(self._dataset_path, cols=cols)
        cls = getattr(importlib.import_module('elliot.splitter'), 'Splitter')
        splitter = cls(self._sample_df, ns)
        return splitter.process_splitting()

    @pytest.mark.parametrize('ratio', [True, False])
    def test_temporal_holdout(self, ratio):
        config = {
            'test_splitting': {
                'strategy': 'temporal_hold_out',
                **({'test_ratio': 0.5} if ratio else {'leave_n_out': 2})
            }
        }

        result = self._apply_splitter(config)

        assert len(result) == 1
        train, test = result[0]
        assert not train.empty and not test.empty
        if ratio:
            assert len(train) + len(test) == len(self._sample_df)
        else:
            assert all(test.groupby('userId').size() == 2)

    @pytest.mark.parametrize('ratio', [True, False])
    def test_random_subsampling(self, ratio):
        config = {
            'test_splitting': {
                'strategy': 'random_subsampling',
                'folds': 3,
                **({'test_ratio': 0.3} if ratio else {'leave_n_out': 2})
            }
        }

        result = self._apply_splitter(config)

        assert len(result) == 3
        for train, test in result:
            assert not train.empty and not test.empty
            assert len(train) + len(test) == len(self._sample_df)

    def test_random_cross_validation(self):
        config = {
            'test_splitting': {
                'strategy': 'random_cross_validation',
                'folds': 3
            }
        }

        result = self._apply_splitter(config)

        assert len(result) == 3
        for train, test in result:
            assert not train.empty and not test.empty
            assert len(train) + len(test) == len(self._sample_df)

    def test_fixed_timestamp(self):
        config = {
            'test_splitting': {
                'strategy': 'fixed_timestamp',
                'timestamp': 5
            }
        }

        result = self._apply_splitter(config)

        assert len(result) == 1
        train, test = result[0]
        assert all(test["timestamp"] >= 5)
        assert all(train["timestamp"] < 5)

    def test_best_timestamp(self):
        config = {
            'test_splitting': {
                'strategy': 'fixed_timestamp',
                'timestamp': 'best',
                'min_below': 1,
                'min_over': 1
            }
        }

        result = self._apply_splitter(config)

        assert len(result) == 1
        train, test = result[0]
        assert not train.empty and not test.empty
        assert train['timestamp'].max() < test['timestamp'].min()

    def test_missing_strategy_raises(self):
        config = {
            'test_splitting': {}
        }
        with pytest.raises(Exception, match="Strategy option not found"):
            self._apply_splitter(config)

    def test_missing_test_splitting(self):
        with pytest.raises(Exception, match="Test splitting strategy is not defined"):
            self._apply_splitter({})

    def test_saving_on_disk(self):
        save_folder = current_path / 'splitting_strategies/splitting/'
        config = {
            'save_on_disk': True,
            'save_folder': str(save_folder),
            'test_splitting': {
                'strategy': 'fixed_timestamp',
                'timestamp': 8
            }
        }

        self._apply_splitter(config)

        assert (save_folder / "0" / "test.tsv").exists()

    def test_train_validation_test_split(self):
        config = {
            'test_splitting': {
                'strategy': 'random_cross_validation',
                'folds': 2
            },
            'validation_splitting': {
                'strategy': 'temporal_hold_out',
                'test_ratio': 0.5
            }
        }

        result = self._apply_splitter(config)

        assert len(result) == 2
        for train_val_list, test in result:
            assert isinstance(train_val_list, list)
            for train, val in train_val_list:
                assert not train.empty
                assert not val.empty


if __name__ == '__main__':
    pytest.main()
