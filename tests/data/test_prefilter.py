import importlib
from tests.params import params_pre_filtering as p
from tests.params import params_pre_filtering_fail as p_fail
from tests.utils import *

import pytest


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
            'strategy': 'global_threshold',
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
            'strategy': 'global_threshold',
            'threshold': 'average'
        }

        df = custom_read_dataset(params['dataset_path'])
        filtered = apply_filter(config, df=df)

        assert filtered["rating"].mean() >= df["rating"].mean()

    @pytest.mark.parametrize('params', p['user_average'])
    def test_user_average(self, params):
        config = {
            'strategy': 'user_average',
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
            'strategy': 'user_k_core',
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
            'strategy': 'item_k_core',
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
            'strategy': 'iterative_k_core',
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
            'strategy': 'n_rounds_k_core',
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
            'strategy': 'cold_users',
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
        assert isinstance(exc_info.value, (ValueError, TypeError))

    @pytest.mark.parametrize('params, th', p_fail['invalid_global_threshold'])
    @time_single_test
    def test_invalid_or_missing_global_threshold(self, params, th):
        config = {
            'strategy': 'global_threshold',
            **({'threshold': th} if th is not None else {})
        }

        self._assert_invalid_config(config, params['dataset_path'])

    @pytest.mark.parametrize('params', p['user_average'])
    def test_user_average_with_extra_param(self, params):
        config = {
            'strategy': 'user_average',
            'threshold': None
        }

        apply_filter(config, params['dataset_path'])

    @pytest.mark.parametrize('params, c', p_fail['invalid_user_k_core'])
    @time_single_test
    def test_invalid_or_missing_user_k_core(self, params, c):
        config = {
            'strategy': 'user_k_core',
            **({'core': c} if c is not None else {})
        }

        self._assert_invalid_config(config, params['dataset_path'])

    @pytest.mark.parametrize('params, c', p_fail['invalid_item_k_core'])
    @time_single_test
    def test_invalid_or_missing_item_k_core(self, params, c):
        config = {
            'strategy': 'item_k_core',
            **({'core': c} if c is not None else {})
        }

        self._assert_invalid_config(config, params['dataset_path'])

    @pytest.mark.parametrize('params, c', p_fail['invalid_iterative_k_core'])
    @time_single_test
    def test_invalid_or_missing_iterative_k_core(self, params, c):
        config = {
            'strategy': 'iterative_k_core',
            **({'core': c} if c is not None else {})
        }

        self._assert_invalid_config(config, params['dataset_path'])

    @pytest.mark.parametrize('params, c, r', p_fail['invalid_n_rounds_combinations'])
    @time_single_test
    def test_invalid_or_missing_rounds_k_core(self, params, c, r):
        config = {
            'strategy': 'n_rounds_k_core',
            **({'core': c} if c is not None else {}),
            **({'rounds': r} if r is not None else {})
        }

        self._assert_invalid_config(config, params['dataset_path'])

    @pytest.mark.parametrize('params, th', p_fail['invalid_cold_users'])
    @time_single_test
    def test_invalid_or_missing_cold_users_threshold(self, params, th):
        config = {
            'strategy': 'cold_users',
            **({'threshold': th} if th is not None else {})
        }

        self._assert_invalid_config(config, params['dataset_path'])


if __name__ == '__main__':
    pytest.main()
