import importlib
from itertools import product
from tests.test_params import params_pre_filtering
from tests.test_utils import *

import pytest


def custom_read_dataset(path):
    path = params_pre_filtering['dataset_path'] if 'dataset_path' in params_pre_filtering.keys() else path
    df = read_dataset(path)
    return df

def apply_filter(config, path=None, df=None):
    ns = create_namespace(config)
    df = custom_read_dataset(path) if df is None else df
    cls = getattr(importlib.import_module('elliot.prefiltering'), 'PreFilter')
    prefilter = cls(df, [ns])
    return prefilter.filter()


class TestPreFilter:

    @pytest.mark.parametrize('params', params_pre_filtering['global_threshold'])
    def test_global_threshold(self, params):
        config = {
            'strategy': 'global_threshold',
            'threshold': params['threshold']
        }

        filtered = apply_filter(config, params['dataset_path'])

        assert all(filtered["rating"] >= params['threshold'])

    @pytest.mark.parametrize('params', params_pre_filtering['global_threshold'])
    def test_global_average(self, params):
        config = {
            'strategy': 'global_threshold',
            'threshold': 'average'
        }

        df = custom_read_dataset(params['dataset_path'])
        filtered = apply_filter(config, df=df)

        assert filtered["rating"].mean() >= df["rating"].mean()

    @pytest.mark.parametrize('params', params_pre_filtering['user_average'])
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

    @pytest.mark.parametrize('params', params_pre_filtering['user_k_core'])
    def test_user_k_core(self, params):
        config = {
            'strategy': 'user_k_core',
            'core': params['core']
        }

        filtered = apply_filter(config, params['dataset_path'])

        assert filtered['userId'].value_counts().min() >= params['core']

    @pytest.mark.parametrize('params', params_pre_filtering['item_k_core'])
    def test_item_k_core(self, params):
        config = {
            'strategy': 'item_k_core',
            'core': params['core']
        }

        filtered = apply_filter(config, params['dataset_path'])

        assert filtered['itemId'].value_counts().min() >= params['core']

    @pytest.mark.parametrize('params', params_pre_filtering['iterative_k_core'])
    def test_iterative_k_core(self, params):
        config = {
            'strategy': 'iterative_k_core',
            'core': params['core']
        }

        filtered = apply_filter(config, params['dataset_path'])

        assert filtered['userId'].value_counts().min() >= params['core']
        assert filtered['itemId'].value_counts().min() >= params['core']

    @pytest.mark.parametrize('params', params_pre_filtering['n_rounds_k_core'])
    def test_n_rounds_k_core(self, params):
        config = {
            'strategy': 'n_rounds_k_core',
            'core': params['core'],
            'rounds': params['rounds']
        }

        filtered = apply_filter(config, params['dataset_path'])

        assert filtered['userId'].value_counts().min() >= params['core']
        assert filtered['itemId'].value_counts().min() >= params['core']

    @pytest.mark.parametrize('params', params_pre_filtering['cold_users'])
    def test_retain_cold_users(self, params):
        config = {
            'strategy': 'cold_users',
            'threshold': params['threshold']
        }

        filtered = apply_filter(config, params['dataset_path'])

        assert filtered['userId'].value_counts().min() <= params['threshold']


class TestPreFilterFailures:

    @pytest.mark.parametrize('params, th', list(product(
        params_pre_filtering['global_threshold'],
        [[3], -3, 'invalid', None]
    )))
    @time_single_test
    @pytest.mark.xfail(raises=(ValueError, TypeError))
    def test_invalid_or_missing_global_threshold(self, params, th):
        config = {
            'strategy': 'global_threshold',
            **({'threshold': th} if th is not None else {})
        }

        apply_filter(config, params['dataset_path'])

    @pytest.mark.parametrize('params', params_pre_filtering['user_average'])
    def test_user_average_with_extra_param(self, params):
        config = {
            'strategy': 'user_average',
            'threshold': None
        }

        apply_filter(config, params['dataset_path'])

    @pytest.mark.parametrize('params, c', list(product(
        params_pre_filtering['user_k_core'],
        [-5, 'abc', None]
    )))
    @time_single_test
    @pytest.mark.xfail(raises=(ValueError, TypeError))
    def test_invalid_or_missing_user_k_core(self, params, c):
        config = {
            'strategy': 'user_k_core',
            **({'core': c} if c is not None else {})
        }

        apply_filter(config, params['dataset_path'])

    @pytest.mark.parametrize('params, c', list(product(
        params_pre_filtering['item_k_core'],
        [-5, 2.5, None]
    )))
    @time_single_test
    @pytest.mark.xfail(raises=(ValueError, TypeError))
    def test_invalid_or_missing_item_k_core(self, params, c):
        config = {
            'strategy': 'item_k_core',
            **({'core': c} if c is not None else {})
        }

        apply_filter(config, params['dataset_path'])

    @pytest.mark.parametrize('params, c', list(product(
        params_pre_filtering['iterative_k_core'],
        [-5, 'x', None]
    )))
    @time_single_test
    @pytest.mark.xfail(raises=(ValueError, TypeError))
    def test_invalid_or_missing_iterative_k_core(self, params, c):
        config = {
            'strategy': 'iterative_k_core',
            **({'core': c} if c is not None else {})
        }

        apply_filter(config, params['dataset_path'])

    @pytest.mark.parametrize('params, c, r', list(product(
        params_pre_filtering['n_rounds_k_core'],
        [2, -5, 'x', None],
        [2, -5, 'y', None]
    )))
    @time_single_test
    @pytest.mark.xfail(raises=(ValueError, TypeError))
    def test_invalid_or_missing_rounds_k_core(self, params, c, r):
        config = {
            'strategy': 'n_rounds_k_core',
            **({'core': c} if c is not None else {}),
            **({'rounds': r} if r is not None else {})
        }

        apply_filter(config, params['dataset_path'])

    @pytest.mark.parametrize('params, th', list(product(
        params_pre_filtering['cold_users'],
        [-99, 'cold', None]
    )))
    @time_single_test
    @pytest.mark.xfail(raises=(ValueError, TypeError))
    def test_invalid_or_missing_cold_users_threshold(self, params, th):
        config = {
            'strategy': 'cold_users',
            **({'threshold': th} if th is not None else {})
        }

        apply_filter(config, params['dataset_path'])


if __name__ == '__main__':
    pytest.main()
