from pathlib import Path
import sys
from utils import *

import pytest

current_path = Path(__file__).resolve().parent
sys.path.append(str(current_path))

from elliot.prefiltering.standard_prefilters import PreFilter


class TestPreFilter:

    _dataset_path = str(current_path / 'prefiltering_strategies/dataset.tsv')

    def _apply_filter(self, config):
        ns = create_namespace(config)
        self._sample_df = read_dataset(self._dataset_path, cols=True)
        filtered = PreFilter.single_filter(self._sample_df, ns)
        return filtered

    def test_global_threshold(self):
        config = {
            'strategy': 'global_threshold',
            'threshold': 4
        }
        filtered = self._apply_filter(config)
        assert all(filtered["rating"] >= 4)

    def test_global_average(self):
        config = {
            'strategy': 'global_threshold',
            'threshold': 'average'
        }
        filtered = self._apply_filter(config)
        assert filtered["rating"].mean() >= self._sample_df["rating"].mean()

    def test_user_average(self):
        config = {
            'strategy': 'user_average',
        }
        filtered = self._apply_filter(config)
        assert set(filtered.columns) == set(self._sample_df.columns)

    def test_user_k_core(self):
        config = {
            'strategy': 'user_k_core',
            'core': 3
        }
        filtered = self._apply_filter(config)
        assert filtered["userId"].nunique() == 1  # only user 3 has 3 ratings

    def test_item_k_core(self):
        config = {
            'strategy': 'item_k_core',
            'core': 10
        }
        filtered = self._apply_filter(config)
        assert all(filtered["itemId"] == 10)

    def test_iterative_k_core(self):
        config = {
            'strategy': 'iterative_k_core',
            'core': 2
        }
        filtered = self._apply_filter(config)
        assert all(filtered.groupby("userId").size() >= 2)

    def test_rounds_k_core(self):
        config = {
            'strategy': 'n_rounds_k_core',
            'core': 2,
            'rounds': 2
        }
        filtered = self._apply_filter(config)
        assert all(filtered.groupby("itemId").size() >= 2)

    def test_cold_users(self,):
        config = {
            'strategy': 'cold_users',
            'threshold': 2
        }
        filtered = self._apply_filter(config)
        assert all(filtered.groupby("userId").size() <= 2)

    def test_missing_strategy_raises(self):
        with pytest.raises(ValueError):
            self._apply_filter({})

    def test_invalid_threshold_value(self):
        config = {
            'strategy': 'global_threshold',
            'threshold': 'invalid'
        }
        with pytest.raises(ValueError):
            self._apply_filter(config)


if __name__ == '__main__':
    pytest.main()
