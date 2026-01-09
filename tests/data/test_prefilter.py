import pytest
from pathlib import Path

from elliot.dataset import DataSetLoader
from elliot.utils.enums import PreFilteringStrategy, DataLoadingStrategy

from tests.params import params_pre_filtering_fail as p
from tests.utils import create_namespace, data_path

current_path = Path(__file__).parent


def load_and_filter_data(config_dict):
    config = {
        "experiment": {
            "data_config": {
                "strategy": DataLoadingStrategy.DATASET.value,
                "data_path": data_path
            },
            **config_dict
        }
    }
    ns = create_namespace(config, current_path)
    loader = DataSetLoader(ns)
    return loader.dataframe


class TestPreFilter:

    def test_global_threshold(self):
        config = {
            "dataset": "filter_ratings_by_global_threshold",
            "prefiltering": {
                "strategy": PreFilteringStrategy.GLOBAL_TH.value,
                "threshold": 3
            }
        }

        filtered = load_and_filter_data(config)

        assert not filtered.empty
        if len(filtered) < 20:
            assert all(filtered["rating"] >= 3)

    def test_global_average(self):
        config = {
            "dataset": "filter_ratings_by_global_threshold",
            "prefiltering": {
                "strategy": PreFilteringStrategy.GLOBAL_TH.value,
            }
        }

        filtered = load_and_filter_data(config)

        assert filtered["rating"].mean() >= 3

    def test_user_average(self):
        config = {
            "dataset": "filter_ratings_by_user_average",
            "prefiltering": {
                "strategy": PreFilteringStrategy.USER_AVG.value,
            }
        }

        filtered = load_and_filter_data(config)

        assert all(filtered["rating"] >= 3)

    def test_user_k_core(self):
        config = {
            "dataset": "filter_user_k_core",
            "prefiltering": {
                "strategy": PreFilteringStrategy.USER_K_CORE.value,
                "core": 2
            }
        }

        filtered = load_and_filter_data(config)

        assert not filtered.empty
        if len(filtered) < 13:
            assert filtered['userId'].value_counts().min() >= 2

    def test_item_k_core(self):
        config = {
            "dataset": "filter_item_k_core",
            "prefiltering": {
                "strategy": PreFilteringStrategy.ITEM_K_CORE.value,
                "core": 3
            }
        }

        filtered = load_and_filter_data(config)

        assert not filtered.empty
        if len(filtered) < 14:
            assert filtered['itemId'].value_counts().min() >= 3

    def test_iterative_k_core(self):
        config = {
            "dataset": "filter_iterative_k_core",
            "prefiltering": {
                "strategy": PreFilteringStrategy.ITER_K_CORE.value,
                "core": 2
            }
        }

        filtered = load_and_filter_data(config)

        assert not filtered.empty
        if len(filtered) < 8:
            assert filtered['userId'].value_counts().min() >= 2
            assert filtered['itemId'].value_counts().min() >= 2

    def test_n_rounds_k_core(self):
        config = {
            "dataset": "filter_n_rounds_k_core",
            "prefiltering": {
                "strategy": PreFilteringStrategy.N_ROUNDS_K_CORE.value,
                "core": 2,
                "rounds": 2
            }
        }

        filtered = load_and_filter_data(config)

        assert not filtered.empty
        if len(filtered) < 9:
            assert filtered['userId'].value_counts().min() >= 2
            assert filtered['itemId'].value_counts().min() >= 2

    def test_retain_cold_users(self):
        config = {
            "dataset": "filter_retain_cold_users",
            "prefiltering": {
                "strategy": PreFilteringStrategy.COLD_USERS.value,
                "threshold": 2
            }
        }

        filtered = load_and_filter_data(config)

        assert not filtered.empty
        if len(filtered) < 13:
            assert filtered['userId'].value_counts().min() <= 2


class TestPreFilterFailures:

    @pytest.mark.parametrize("params", p["invalid_global_threshold"])
    def test_invalid_or_missing_params_global_threshold(self, params):
        config = {
            "dataset": "filter_ratings_by_global_threshold",
            "prefiltering": {
                "strategy": PreFilteringStrategy.GLOBAL_TH.value,
                "threshold": params["threshold"]
            }
        }

        with pytest.raises(ValueError):
            load_and_filter_data(config)

    def test_user_average_with_extra_param(self):
        config = {
            "dataset": "filter_ratings_by_user_average",
            "prefiltering": {
                "strategy": PreFilteringStrategy.USER_AVG.value,
                "threshold": None
            }
        }

        load_and_filter_data(config)

    @pytest.mark.parametrize("params", p["invalid_user_k_core"])
    def test_invalid_or_missing_params_user_k_core(self, params):
        config = {
            "dataset": "filter_user_k_core",
            "prefiltering": {
                "strategy": PreFilteringStrategy.USER_K_CORE.value,
                **({"core": params["core"]} if params["core"] is not None else {})
            }
        }

        with pytest.raises(ValueError):
            load_and_filter_data(config)

    @pytest.mark.parametrize("params", p["invalid_item_k_core"])
    def test_invalid_or_missing_params_item_k_core(self, params):
        config = {
            "dataset": "filter_item_k_core",
            "prefiltering": {
                "strategy": PreFilteringStrategy.ITEM_K_CORE.value,
                **({"core": params["core"]} if params["core"] is not None else {})
            }
        }

        with pytest.raises(ValueError):
            load_and_filter_data(config)

    @pytest.mark.parametrize("params", p["invalid_iterative_k_core"])
    def test_invalid_or_missing_params_iterative_k_core(self, params):
        config = {
            "dataset": "filter_iterative_k_core",
            "prefiltering": {
                "strategy": PreFilteringStrategy.ITER_K_CORE.value,
                **({"core": params["core"]} if params["core"] is not None else {})
            }
        }

        with pytest.raises(ValueError):
            load_and_filter_data(config)

    @pytest.mark.parametrize("params", p["invalid_n_rounds_combinations"])
    def test_invalid_or_missing_params_rounds_k_core(self, params):
        if params["core"] == 2 and params["rounds"] == 2:
            pytest.skip("Test requires at least one invalid parameter to be meaningful.")

        config = {
            "dataset": "filter_n_rounds_k_core",
            "prefiltering": {
                "strategy": PreFilteringStrategy.N_ROUNDS_K_CORE.value,
                "core": params["core"],
                "rounds": params["rounds"]
            }
        }

        with pytest.raises(ValueError):
            load_and_filter_data(config)

    @pytest.mark.parametrize("params", p["invalid_cold_users"])
    def test_invalid_or_missing_params_cold_users_threshold(self, params):
        config = {
            "dataset": "filter_retain_cold_users",
            "prefiltering": {
                "strategy": PreFilteringStrategy.COLD_USERS.value,
                **({"threshold": params["threshold"]} if params["threshold"] is not None else {})
            }
        }

        with pytest.raises((ValueError, AttributeError)):
            load_and_filter_data(config)


if __name__ == '__main__':
    pytest.main()
