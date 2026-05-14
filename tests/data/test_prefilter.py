import pytest

from elliot.dataset import DataSetLoader
from elliot.namespace import build_namespace
from elliot.utils.folder import path_joiner

from tests.params import params_pre_filtering_fail as p, dataset_path

current_path = path_joiner(__file__)


def load_data(config_dict):
    config_data = {
        "experiment": {
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path,
                "reader": {"header": True}
            },
            **config_dict
        }
    }
    config = build_namespace(config_path=current_path, config_data=config_data)
    loader = DataSetLoader(config=config)
    return loader.dataframe


class TestPreFilter:

    def test_global_threshold(self):
        config = {
            "dataset": "filter_ratings_by_global_threshold",
            "prefiltering": {
                "strategy": "global_threshold",
                "threshold": 3
            }
        }

        filtered = load_data(config)

        assert not filtered.empty
        if len(filtered) < 20:
            assert all(filtered["rating"] >= 3)

    def test_global_average(self):
        config = {
            "dataset": "filter_ratings_by_global_threshold",
            "prefiltering": {
                "strategy": "global_threshold",
            }
        }

        filtered = load_data(config)

        assert filtered["rating"].mean() >= 3

    def test_user_average(self):
        config = {
            "dataset": "filter_ratings_by_user_average",
            "prefiltering": {
                "strategy": "user_average",
            }
        }

        filtered = load_data(config)

        assert all(filtered["rating"] >= 3)

    def test_user_k_core(self):
        config = {
            "dataset": "filter_user_k_core",
            "prefiltering": {
                "strategy": "user_k_core",
                "core": 2
            }
        }

        filtered = load_data(config)

        assert not filtered.empty
        if len(filtered) < 13:
            assert filtered['userId'].value_counts().min() >= 2

    def test_item_k_core(self):
        config = {
            "dataset": "filter_item_k_core",
            "prefiltering": {
                "strategy": "item_k_core",
                "core": 3
            }
        }

        filtered = load_data(config)

        assert not filtered.empty
        if len(filtered) < 14:
            assert filtered['itemId'].value_counts().min() >= 3

    def test_iterative_k_core(self):
        config = {
            "dataset": "filter_iterative_k_core",
            "prefiltering": {
                "strategy": "iterative_k_core",
                "core": 2
            }
        }

        filtered = load_data(config)

        assert not filtered.empty
        if len(filtered) < 8:
            assert filtered['userId'].value_counts().min() >= 2
            assert filtered['itemId'].value_counts().min() >= 2

    def test_n_rounds_k_core(self):
        config = {
            "dataset": "filter_n_rounds_k_core",
            "prefiltering": {
                "strategy": "n_rounds_k_core",
                "core": 2,
                "rounds": 2
            }
        }

        filtered = load_data(config)

        assert not filtered.empty
        if len(filtered) < 9:
            assert filtered['userId'].value_counts().min() >= 2
            assert filtered['itemId'].value_counts().min() >= 2

    def test_retain_cold_users(self):
        config = {
            "dataset": "filter_retain_cold_users",
            "prefiltering": {
                "strategy": "cold_users",
                "threshold": 2
            }
        }

        filtered = load_data(config)

        assert not filtered.empty
        if len(filtered) < 13:
            assert filtered['userId'].value_counts().min() <= 2


class TestPreFilterFailures:

    @pytest.mark.parametrize("params", p["invalid_global_threshold"])
    def test_invalid_or_missing_params_global_threshold(self, params):
        config = {
            "dataset": "filter_ratings_by_global_threshold",
            "prefiltering": {
                "strategy": "global_threshold",
                "threshold": params["threshold"]
            }
        }

        with pytest.raises(ValueError):
            load_data(config)

    def test_user_average_with_extra_param(self):
        config = {
            "dataset": "filter_ratings_by_user_average",
            "prefiltering": {
                "strategy": "user_average",
                "threshold": None
            }
        }

        load_data(config)

    @pytest.mark.parametrize("params", p["invalid_user_k_core"])
    def test_invalid_or_missing_params_user_k_core(self, params):
        config = {
            "dataset": "filter_user_k_core",
            "prefiltering": {
                "strategy": "user_k_core",
                **({"core": params["core"]} if params["core"] is not None else {})
            }
        }

        with pytest.raises(ValueError):
            load_data(config)

    @pytest.mark.parametrize("params", p["invalid_item_k_core"])
    def test_invalid_or_missing_params_item_k_core(self, params):
        config = {
            "dataset": "filter_item_k_core",
            "prefiltering": {
                "strategy": "item_k_core",
                **({"core": params["core"]} if params["core"] is not None else {})
            }
        }

        with pytest.raises(ValueError):
            load_data(config)

    @pytest.mark.parametrize("params", p["invalid_iterative_k_core"])
    def test_invalid_or_missing_params_iterative_k_core(self, params):
        config = {
            "dataset": "filter_iterative_k_core",
            "prefiltering": {
                "strategy": "iterative_k_core",
                **({"core": params["core"]} if params["core"] is not None else {})
            }
        }

        with pytest.raises(ValueError):
            load_data(config)

    @pytest.mark.parametrize("params", p["invalid_n_rounds_combinations"])
    def test_invalid_or_missing_params_rounds_k_core(self, params):
        if params["core"] == 2 and params["rounds"] == 2:
            pytest.skip("Test requires at least one invalid parameter to be meaningful.")

        config = {
            "dataset": "filter_n_rounds_k_core",
            "prefiltering": {
                "strategy": "n_rounds_k_core",
                "core": params["core"],
                "rounds": params["rounds"]
            }
        }

        with pytest.raises(ValueError):
            load_data(config)

    @pytest.mark.parametrize("params", p["invalid_cold_users"])
    def test_invalid_or_missing_params_cold_users_threshold(self, params):
        config = {
            "dataset": "filter_retain_cold_users",
            "prefiltering": {
                "strategy": "cold_users",
                **({"threshold": params["threshold"]} if params["threshold"] is not None else {})
            }
        }

        with pytest.raises((ValueError, AttributeError)):
            load_data(config)

    @pytest.mark.parametrize("params", p["invalid_strategy"])
    def test_invalid_or_missing_strategy(self, params):
        config = {
            "dataset": "filter_retain_cold_users",
            "prefiltering": {
                **({"strategy": params["strategy"]} if params["strategy"] is not None else {}),
                "threshold": 2
            }
        }

        with pytest.raises(ValueError):
            load_data(config)


if __name__ == '__main__':
    pytest.main()
