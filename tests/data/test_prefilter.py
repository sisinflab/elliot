import pytest
import importlib
from tests.params import params_pre_filtering as p
from tests.params import params_pre_filtering_fail as p_fail
from tests.utils import *

strategy_enum = getattr(importlib.import_module('elliot.utils.enums'), 'PreFilteringStrategy')


def custom_read_dataset(path):
    path = p['dataset_path'] if 'dataset_path' in p.keys() else path
    df = read_dataset(path)
    return df

def apply_filter(config, path=None, df=None):
    ns = create_namespace(config)
    df = custom_read_dataset(path) if df is None else df
    cls = getattr(importlib.import_module('elliot.prefiltering'), 'PreFilter')
    prefilter = cls(df, [ns])
    return prefilter.filter()


class TestPreFilter:

    @pytest.mark.parametrize('params', p['global_threshold'])
    def test_global_threshold(self, params):
        config = {
            'strategy': strategy_enum.GLOBAL_TH.value,
            'threshold': params['threshold']
        }

        df = custom_read_dataset(params['dataset_path'])
        filtered = apply_filter(config, df=df)

        assert not filtered.empty
        if len(filtered) < len(df):
            assert all(filtered["rating"] >= params['threshold'])

    @pytest.mark.parametrize('params', p['global_threshold'])
    def test_global_average(self, params):
        config = {
            'strategy': strategy_enum.GLOBAL_TH.value
        }

        df = custom_read_dataset(params['dataset_path'])
        filtered = apply_filter(config, df=df)

        assert filtered["rating"].mean() >= df["rating"].mean()

    @pytest.mark.parametrize('params', p['user_average'])
    def test_user_average(self, params):
        config = {
            'strategy': strategy_enum.USER_AVG.value,
        }

        df = custom_read_dataset(params['dataset_path'])
        filtered = apply_filter(config, df=df)
        grouped = df.groupby("userId")["rating"].transform("mean")
        mask = df["rating"] >= grouped
        expected = df[mask]

        assert filtered.equals(expected)

    @pytest.mark.parametrize('params', p['user_k_core'])
    def test_user_k_core(self, params):
        config = {
            'strategy': strategy_enum.USER_K_CORE.value,
            'core': params['core']
        }

        df = custom_read_dataset(params['dataset_path'])
        filtered = apply_filter(config, df=df)

        assert not filtered.empty
        if len(filtered) < len(df):
            assert filtered['userId'].value_counts().min() >= params['core']

    @pytest.mark.parametrize('params', p['item_k_core'])
    def test_item_k_core(self, params):
        config = {
            'strategy': strategy_enum.ITEM_K_CORE.value,
            'core': params['core']
        }

        df = custom_read_dataset(params['dataset_path'])
        filtered = apply_filter(config, df=df)

        assert not filtered.empty
        if len(filtered) < len(df):
            assert filtered['itemId'].value_counts().min() >= params['core']

    @pytest.mark.parametrize('params', p['iterative_k_core'])
    def test_iterative_k_core(self, params):
        config = {
            'strategy': strategy_enum.ITER_K_CORE.value,
            'core': params['core']
        }

        df = custom_read_dataset(params['dataset_path'])
        filtered = apply_filter(config, df=df)

        assert not filtered.empty
        if len(filtered) < len(df):
            assert filtered['userId'].value_counts().min() >= params['core']
            assert filtered['itemId'].value_counts().min() >= params['core']

    @pytest.mark.parametrize('params', p['n_rounds_k_core'])
    def test_n_rounds_k_core(self, params):
        config = {
            'strategy': strategy_enum.N_ROUNDS_K_CORE.value,
            'core': params['core'],
            'rounds': params['rounds']
        }

        df = custom_read_dataset(params['dataset_path'])
        filtered = apply_filter(config, df=df)

        assert not filtered.empty
        if len(filtered) < len(df):
            assert filtered['userId'].value_counts().min() >= params['core']
            assert filtered['itemId'].value_counts().min() >= params['core']

    @pytest.mark.parametrize('params', p['cold_users'])
    def test_retain_cold_users(self, params):
        config = {
            'strategy': strategy_enum.COLD_USERS.value,
            'threshold': params['threshold']
        }

        df = custom_read_dataset(params['dataset_path'])
        filtered = apply_filter(config, df=df)

        assert not filtered.empty
        if len(filtered) < len(df):
            assert filtered['userId'].value_counts().min() <= params['threshold']


class TestPreFilterFailures:

    def _assert_invalid_config(self, config, dataset_path):
        with pytest.raises(Exception) as exc_info:
            apply_filter(config, dataset_path)
        assert isinstance(exc_info.value, (AttributeError, TypeError, ValueError))

    @pytest.mark.parametrize('params', p_fail['invalid_global_threshold'])
    @time_single_test
    def test_invalid_or_missing_global_threshold(self, params):
        config = {
            'strategy': strategy_enum.GLOBAL_TH.value,
            **({'threshold': params['threshold']})
        }

        self._assert_invalid_config(config, params['dataset_path'])

    @pytest.mark.parametrize('params', p['user_average'])
    def test_user_average_with_extra_param(self, params):
        config = {
            'strategy': strategy_enum.USER_AVG.value,
            'threshold': None
        }

        apply_filter(config, params['dataset_path'])

    @pytest.mark.parametrize('params', p_fail['invalid_user_k_core'])
    @time_single_test
    def test_invalid_or_missing_user_k_core(self, params):
        config = {
            'strategy': strategy_enum.USER_K_CORE.value,
            **({'core': params['core']} if params['core'] is not None else {})
        }

        self._assert_invalid_config(config, params['dataset_path'])

    @pytest.mark.parametrize('params', p_fail['invalid_item_k_core'])
    @time_single_test
    def test_invalid_or_missing_item_k_core(self, params):
        config = {
            'strategy': strategy_enum.ITEM_K_CORE.value,
            **({'core': params['core']} if params['core'] is not None else {})
        }

        self._assert_invalid_config(config, params['dataset_path'])

    @pytest.mark.parametrize('params', p_fail['invalid_iterative_k_core'])
    @time_single_test
    def test_invalid_or_missing_iterative_k_core(self, params):
        config = {
            'strategy': strategy_enum.ITER_K_CORE.value,
            **({'core': params['core']} if params['core'] is not None else {})
        }

        self._assert_invalid_config(config, params['dataset_path'])

    @pytest.mark.parametrize('params', p_fail['invalid_n_rounds_combinations'])
    @time_single_test
    def test_invalid_or_missing_rounds_k_core(self, params):
        if params['core'] == 2 and params['rounds'] == 2:
            pytest.skip("Test requires at least one invalid parameter to be meaningful.")

        config = {
            'strategy': strategy_enum.N_ROUNDS_K_CORE.value,
            **({'core': params['core']} if params['core'] is not None else {}),
            **({'rounds': params['rounds']} if params['rounds'] is not None else {})
        }

        self._assert_invalid_config(config, params['dataset_path'])

    @pytest.mark.parametrize('params', p_fail['invalid_cold_users'])
    @time_single_test
    def test_invalid_or_missing_cold_users_threshold(self, params):
        config = {
            'strategy': strategy_enum.COLD_USERS.value,
            **({'threshold': params['threshold']} if params['threshold'] is not None else {})
        }

        self._assert_invalid_config(config, params['dataset_path'])


if __name__ == '__main__':
    pytest.main()
