import pytest
from pathlib import Path

from elliot.dataset import DataSetLoader
from elliot.utils.enums import SplittingStrategy, DataLoadingStrategy

from tests.params import params_splitting_fail as p
from tests.utils import create_namespace, data_path

current_path = Path(__file__).parent


def load_and_split_data(config_dict):
    config = {
        "experiment": {
            "dataset": "splitting_strategies",
            "data_config": {
                "strategy": DataLoadingStrategy.DATASET.value,
                "data_path": data_path
            },
            "splitting": {
                **config_dict
            }
        }
    }
    ns = create_namespace(config, current_path)
    loader = DataSetLoader(ns)
    data_list = loader.build()
    return data_list


class TestSplitter:

    def test_temporal_holdout_test_ratio(self):
        config = {
            "test_splitting": {
                "strategy": SplittingStrategy.TEMP_HOLDOUT.value,
                "test_ratio": 0.1
            }
        }

        data = load_and_split_data(config)

        assert len(data) == 1
        train, test = data[0][0].interactions
        assert not train.empty and not test.empty
        assert len(train) + len(test) == 30

    def test_temporal_holdout_leave_n_out(self):
        config = {
            "test_splitting": {
                "strategy": SplittingStrategy.TEMP_HOLDOUT.value,
                "leave_n_out": 3
            }
        }

        data = load_and_split_data(config)

        assert len(data) == 1
        train, test = data[0][0].interactions
        assert not train.empty and not test.empty
        assert all(test.groupby('userId').size() <= 3)

    def test_random_subsampling_test_ratio(self):
        config = {
            "test_splitting": {
                "strategy": SplittingStrategy.RAND_SUB_SMP.value,
                "folds": 10,
                "test_ratio": 0.1
            }
        }

        data = load_and_split_data(config)

        assert len(data) == 10
        inter = [d[0].interactions for d in data]
        for train, test in inter:
            assert not train.empty and not test.empty
            assert len(train) + len(test) == 30

    def test_random_subsampling_leave_n_out(self):
        config = {
            "test_splitting": {
                "strategy": SplittingStrategy.RAND_SUB_SMP.value,
                "folds": 3,
                "leave_n_out": 2
            }
        }

        data = load_and_split_data(config)

        assert len(data) == 3
        inter = [d[0].interactions for d in data]
        for train, test in inter:
            assert not train.empty and not test.empty
            assert len(train) + len(test) == 30

    def test_random_cross_validation(self):
        config = {
            "test_splitting": {
                "strategy": SplittingStrategy.RAND_CV.value,
                "folds": 10,
            }
        }

        data = load_and_split_data(config)

        assert len(data) == 10
        inter = [d[0].interactions for d in data]
        for train, test in inter:
            assert not train.empty and not test.empty
            assert len(train) + len(test) == 30

    def test_fixed_timestamp(self):
        config = {
            "test_splitting": {
                "strategy": SplittingStrategy.FIXED_TS.value,
                "timestamp": 7
            }
        }

        data = load_and_split_data(config)

        assert len(data) == 1
        train, test = data[0][0].interactions
        assert not train.empty and not test.empty
        assert all(test["timestamp"] >= 7)
        assert all(train["timestamp"] < 7)

    def test_best_timestamp(self):
        config = {
            "test_splitting": {
                "strategy": SplittingStrategy.FIXED_TS.value,
                "min_below": 1,
                "min_over": 1
            }
        }

        data = load_and_split_data(config)

        assert len(data) == 1
        train, test = data[0][0].interactions
        assert not train.empty and not test.empty
        assert train['timestamp'].max() < test['timestamp'].min()

    def test_saving_on_disk(self):
        save_folder = "./splitting_strategies/splitting"
        config = {
            "save_on_disk": True,
            "save_folder": save_folder,
            "test_splitting": {
                "strategy": SplittingStrategy.FIXED_TS.value,
                "timestamp": 8
            }
        }

        load_and_split_data(config)

        assert (current_path / Path(save_folder) / "0" / "train.tsv").resolve().exists()
        assert (current_path / Path(save_folder) / "0" / "test.tsv").resolve().exists()

    def test_train_validation_test_split(self):
        config = {
            "test_splitting": {
                "strategy": SplittingStrategy.RAND_CV.value,
                "folds": 3
            },
            "validation_splitting": {
                "strategy": SplittingStrategy.TEMP_HOLDOUT.value,
                "test_ratio": 0.1
            }
        }

        data = load_and_split_data(config)

        assert len(data) == 3
        inter = [d[0].interactions for d in data]
        for train, val, test in inter:
            assert not train.empty
            assert not val.empty
            assert not test.empty


class TestSplitterFailures:

    @pytest.mark.parametrize("params", p["invalid_temporal_holdout_test_ratio"])
    def test_invalid_or_missing_params_temporal_holdout_test_ratio(self, params):
        config = {
            "test_splitting": {
                "strategy": SplittingStrategy.TEMP_HOLDOUT.value,
                **params
            }
        }

        with pytest.raises((ValueError, AttributeError)):
            load_and_split_data(config)

    @pytest.mark.parametrize("params", p["invalid_temporal_holdout_leave_n_out"])
    def test_invalid_or_missing_params_temporal_holdout_leave_n_out(self, params):
        config = {
            "test_splitting": {
                "strategy": SplittingStrategy.TEMP_HOLDOUT.value,
                **params
            }
        }

        with pytest.raises((ValueError, AttributeError)):
            load_and_split_data(config)

    @pytest.mark.parametrize("params", p["invalid_random_subsampling_test_ratio"])
    def test_invalid_or_missing_params_random_subsampling_test_ratio(self, params):
        if params["folds"] == 3 and params.get("test_ratio") == 0.1:
            pytest.skip("Test requires at least one invalid parameter to be meaningful.")

        config = {
            "test_splitting": {
                "strategy": SplittingStrategy.RAND_SUB_SMP.value,
                **params
            }
        }

        with pytest.raises((ValueError, AttributeError)):
            load_and_split_data(config)

    @pytest.mark.parametrize("params", p["invalid_random_subsampling_leave_n_out"])
    def test_invalid_or_missing_params_random_subsampling_leave_n_out(self, params):
        if params["folds"] == 3 and params.get("leave_n_out") == 2:
            pytest.skip("Test requires at least one invalid parameter to be meaningful.")

        config = {
            "test_splitting": {
                "strategy": SplittingStrategy.RAND_SUB_SMP.value,
                **params
            }
        }

        with pytest.raises((ValueError, AttributeError)):
            load_and_split_data(config)

    @pytest.mark.parametrize("params", p["invalid_random_cross_validation"])
    def test_invalid_or_missing_params_random_cross_validation(self, params):
        config = {
            "test_splitting": {
                "strategy": SplittingStrategy.RAND_CV.value,
                **params
            }
        }

        with pytest.raises(ValueError):
            load_and_split_data(config)

    @pytest.mark.parametrize("params", p["invalid_fixed_timestamp"])
    def test_invalid_or_missing_params_fixed_timestamp(self, params):
        config = {
            "test_splitting": {
                "strategy": SplittingStrategy.FIXED_TS.value,
                **params
            }
        }

        with pytest.raises(ValueError):
            load_and_split_data(config)

    @pytest.mark.parametrize("params", p["invalid_best_timestamp"])
    def test_invalid_or_missing_params_best_timestamp(self, params):
        if params["min_below"] == 1 and params["min_over"] == 1:
            pytest.skip("Test requires at least one invalid parameter to be meaningful.")

        config = {
            "test_splitting": {
                "strategy": SplittingStrategy.FIXED_TS.value,
                **params
            }
        }

        with pytest.raises(ValueError):
            load_and_split_data(config)

    @pytest.mark.parametrize("params", p["invalid_strategy"])
    def test_invalid_or_missing_strategy(self, params):
        config = {
            "test_splitting": {
                **params,
                "test_ratio": 0.1
            }
        }

        with pytest.raises(ValueError):
            load_and_split_data(config)

    def test_missing_test_splitting(self):
        config = {}

        with pytest.raises(ValueError):
            load_and_split_data(config)

    def test_invalid_save_folder(self):
        config = {
            "save_on_disk": True,
            "save_folder": 3,
            "test_splitting": {
                "strategy": SplittingStrategy.FIXED_TS.value,
                "timestamp": 8
            }
        }

        with pytest.raises(ValueError):
            load_and_split_data(config)


if __name__ == '__main__':
    pytest.main()
