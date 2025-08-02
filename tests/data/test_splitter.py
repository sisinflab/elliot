import pytest
import importlib
from tests.params import params_splitting as p
from tests.params import params_splitting_fail as p_fail
from tests.utils import *


def custom_read_dataset():
    cols = ['userId', 'timestamp']
    df = read_dataset(p['dataset_path'], cols=cols)
    return df

def apply_splitter(config, df=None):
    ns = create_namespace(config)
    df = custom_read_dataset() if df is None else df
    cls = getattr(importlib.import_module('elliot.splitter'), 'Splitter')
    splitter = cls(df, ns)
    return splitter.process_splitting()


class TestSplitter:

    @pytest.mark.parametrize('params', p['temporal_hold_out'])
    @time_single_test
    def test_temporal_holdout(self, params):
        config = {
            'test_splitting': {
                'strategy': 'temporal_hold_out',
                **{k: v for k, v in params.items() if k in ('test_ratio', 'leave_n_out') and v is not None}
            }
        }

        df = custom_read_dataset()
        result = apply_splitter(config, df)

        assert len(result) == 1
        train, test = result[0]
        assert not train.empty and not test.empty
        if 'test_ratio' in params:
            assert len(train) + len(test) == len(df)
        else:
            assert all(test.groupby('userId').size() <= params['leave_n_out'])

    @pytest.mark.parametrize('params', p['random_subsampling'])
    @time_single_test
    def test_random_subsampling(self, params):
        config = {
            'test_splitting': {
                'strategy': 'random_subsampling',
                'folds': params['folds'],
                **{k: v for k, v in params.items() if k in ('test_ratio', 'leave_n_out') and v is not None}
            }
        }

        df = custom_read_dataset()
        result = apply_splitter(config, df)

        assert len(result) == params['folds']
        for train, test in result:
            assert not train.empty and not test.empty
            assert len(train) + len(test) == len(df)

    @pytest.mark.parametrize('params', p['random_cross_validation'])
    def test_random_cross_validation(self, params):
        config = {
            'test_splitting': {
                'strategy': 'random_cross_validation',
                'folds': params['folds']
            }
        }

        df = custom_read_dataset()
        result = apply_splitter(config, df)

        assert len(result) == params['folds']
        for train, test in result:
            assert not train.empty and not test.empty
            assert len(train) + len(test) == len(df)

    @pytest.mark.parametrize('params', p['fixed_timestamp'])
    @time_single_test
    def test_fixed_timestamp(self, params):
        config = {
            'test_splitting': {
                'strategy': 'fixed_timestamp',
                'timestamp': params['timestamp'] if 'timestamp' in params else 'best',
                **{k: v for k, v in params.items() if k in ('min_below', 'min_over') and v is not None}
            }
        }

        df = custom_read_dataset()
        result = apply_splitter(config, df)

        assert len(result) == 1
        train, test = result[0]
        assert not train.empty and not test.empty
        if isinstance('timestamp', int):
            assert all(test["timestamp"] >= params['timestamp'])
            assert all(train["timestamp"] < params['timestamp'])
        else:
            assert train['timestamp'].max() < test['timestamp'].min()

    def test_saving_on_disk(self):
        config = {
            'save_on_disk': True,
            'save_folder': p['save_folder'],
            'test_splitting': {
                'strategy': 'fixed_timestamp',
                'timestamp': 8
            }
        }

        apply_splitter(config)
        assert (Path(p['save_folder']) / "0" / "test.tsv").exists()

    def test_train_validation_test_split(self):
        config = {
            'test_splitting': {
                'strategy': 'random_cross_validation',
                'folds': 3
            },
            'validation_splitting': {
                'strategy': 'temporal_hold_out',
                'test_ratio': 0.1
            }
        }

        result = apply_splitter(config)

        assert len(result) == 3
        for train_val_list, test in result:
            assert isinstance(train_val_list, list)
            for train, val in train_val_list:
                assert not train.empty
                assert not val.empty


class TestSplitterFailures:

    def _assert_invalid_config(self, config):
        with pytest.raises(Exception) as exc_info:
            apply_splitter(config)
        assert isinstance(exc_info.value, (AttributeError, TypeError, ValueError))

    @pytest.mark.parametrize('params', p_fail['invalid_temporal_hold_out'])
    @time_single_test
    def test_invalid_or_missing_temporal_holdout(self, params):
        config = {
            'test_splitting': {
                'strategy': 'temporal_hold_out',
                **{k: v for k, v in params.items() if k in ('test_ratio', 'leave_n_out') and v is not None}
            }
        }

        self._assert_invalid_config(config)

    @pytest.mark.parametrize('params', p_fail['invalid_random_subsampling'])
    @time_single_test
    def test_invalid_or_missing_random_subsampling(self, params):
        if params['folds'] == 3 and (params.get('test_ratio') == 0.1 or params.get('leave_n_out') == 2):
            pytest.skip("Test requires at least one invalid parameter to be meaningful.")

        config = {
            'test_splitting': {
                'strategy': 'random_subsampling',
                'folds': params['folds'],
                **{k: v for k, v in params.items() if k in ('test_ratio', 'leave_n_out') and v is not None}
            }
        }

        self._assert_invalid_config(config)

    @pytest.mark.parametrize('params', p_fail['invalid_random_cross_validation'])
    def test_invalid_or_missing_random_cross_validation(self, params):
        config = {
            'test_splitting': {
                'strategy': 'random_cross_validation',
                **({'folds': params['folds']} if params['folds'] is not None else {})
            }
        }

        self._assert_invalid_config(config)

    @pytest.mark.parametrize('params', p_fail['invalid_fixed_timestamp'])
    @time_single_test
    def test_invalid_or_missing_fixed_timestamp(self, params):
        if params.get('min_below') == 1 and params.get('min_over') == 1:
            pytest.skip("Test requires at least one invalid parameter to be meaningful.")

        timestamp_config = (
            {'timestamp': params['timestamp']} if params.get('timestamp') is not None
            else {'timestamp': 'best'} if 'timestamp' not in params
            else {}
        )

        config = {
            'test_splitting': {
                'strategy': 'fixed_timestamp',
                **timestamp_config,
                **{k: v for k, v in params.items() if k in ('min_below', 'min_over') and v is not None}
            }
        }

        self._assert_invalid_config(config)

    def test_missing_strategy(self):
        config = {
            'test_splitting': {}
        }

        self._assert_invalid_config(config)

    def test_missing_test_splitting(self):
        config = {}

        self._assert_invalid_config(config)

    def test_missing_save_folder(self):
        config = {
            'save_on_disk': True,
            'test_splitting': {
                'strategy': 'fixed_timestamp',
                'timestamp': 8
            }
        }

        self._assert_invalid_config(config)


if __name__ == '__main__':
    pytest.main()
