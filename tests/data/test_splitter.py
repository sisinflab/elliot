import pytest

from elliot.dataset import DataSetLoader
from elliot.namespace import build_namespace
from elliot.utils.folder import path_joiner, check_path, parent_dir

from tests.params import params_splitting_fail as p
from tests.utils import dataset_path

current_path = path_joiner(__file__)


def load_and_split_data(config_dict):
    config_data = {
        "experiment": {
            "dataset": "splitting_strategies",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path,
                "reader": {"header": True}
            },
            "splitting": {
                **config_dict
            }
        }
    }
    config = build_namespace(config_path=current_path, config_data=config_data)
    loader = DataSetLoader(config=config)
    val_data, main_data = loader.build()
    return val_data, main_data


class TestSplitter:

    def test_temporal_holdout_test_ratio(self):
        config = {
            "test_splitting": {
                "strategy": "temporal_holdout",
                "test_ratio": 0.1
            }
        }

        _, train_test = load_and_split_data(config)

        assert len(train_test) == 1
        train = train_test[0].train_set.dataframe
        test = train_test[0].eval_set.dataframe
        assert not train.empty and not test.empty
        assert len(train) + len(test) == 30

    def test_temporal_holdout_leave_n_out(self):
        config = {
            "test_splitting": {
                "strategy": "temporal_holdout",
                "leave_n_out": 3
            }
        }

        _, train_test = load_and_split_data(config)

        assert len(train_test) == 1
        train = train_test[0].train_set.dataframe
        test = train_test[0].eval_set.dataframe
        assert not train.empty and not test.empty
        assert all(test.groupby('userId').size() <= 3)

    def test_random_subsampling_test_ratio(self):
        config = {
            "test_splitting": {
                "strategy": "random_subsampling",
                "folds": 10,
                "test_ratio": 0.1
            }
        }

        _, train_test = load_and_split_data(config)

        assert len(train_test) == 10
        train_list = [t.train_set.dataframe for t in train_test]
        test_list = [t.eval_set.dataframe for t in train_test]
        for train, test in zip(train_list, test_list):
            assert not train.empty and not test.empty
            assert len(train) + len(test) == 30

    def test_random_subsampling_leave_n_out(self):
        config = {
            "test_splitting": {
                "strategy": "random_subsampling",
                "folds": 3,
                "leave_n_out": 2
            }
        }

        _, train_test = load_and_split_data(config)

        assert len(train_test) == 3
        train_list = [t.train_set.dataframe for t in train_test]
        test_list = [t.eval_set.dataframe for t in train_test]
        for train, test in zip(train_list, test_list):
            assert not train.empty and not test.empty
            assert len(train) + len(test) == 30

    def test_random_cross_validation(self):
        config = {
            "test_splitting": {
                "strategy": "random_cross_validation",
                "folds": 10,
            }
        }

        _, train_test = load_and_split_data(config)

        assert len(train_test) == 10
        train_list = [t.train_set.dataframe for t in train_test]
        test_list = [t.eval_set.dataframe for t in train_test]
        for train, test in zip(train_list, test_list):
            assert not train.empty and not test.empty
            assert len(train) + len(test) == 30

    def test_fixed_timestamp(self):
        config = {
            "test_splitting": {
                "strategy": "fixed_timestamp",
                "timestamp": 7
            }
        }

        _, train_test = load_and_split_data(config)

        assert len(train_test) == 1
        train = train_test[0].train_set.dataframe
        test = train_test[0].eval_set.dataframe
        assert not train.empty and not test.empty
        assert all(test["timestamp"] >= 7)
        assert all(train["timestamp"] < 7)

    def test_best_timestamp(self):
        config = {
            "test_splitting": {
                "strategy": "fixed_timestamp",
                "min_below": 1,
                "min_over": 1
            }
        }

        _, train_test = load_and_split_data(config)

        assert len(train_test) == 1
        train = train_test[0].train_set.dataframe
        test = train_test[0].eval_set.dataframe
        assert not train.empty and not test.empty
        assert train['timestamp'].max() < test['timestamp'].min()

    def test_saving_on_disk(self):
        save_folder = "./splitting_strategies/splitting"
        config = {
            "save_on_disk": True,
            "save_folder": save_folder,
            "test_splitting": {
                "strategy": "fixed_timestamp",
                "timestamp": 8
            }
        }

        load_and_split_data(config)

        current_folder = parent_dir(current_path)
        train_path = path_joiner(current_folder, save_folder, "0", "train.tsv")
        test_path = path_joiner(current_folder, save_folder, "0", "test.tsv")
        assert check_path(train_path)
        assert check_path(test_path)

    def test_train_validation_test_split(self):
        config = {
            "test_splitting": {
                "strategy": "random_cross_validation",
                "folds": 3
            },
            "validation_splitting": {
                "strategy": "temporal_holdout",
                "test_ratio": 0.1
            }
        }

        train_val, train_test = load_and_split_data(config)

        assert len(train_val) == 3
        train_list = [t[0].train_set.dataframe for t in train_val]
        val_list = [t[0].eval_set.dataframe for t in train_val]
        for train, val in zip(train_list, val_list):
            assert not train.empty and not val.empty

        assert len(train_test) == 3
        train_list = [t.train_set.dataframe for t in train_test]
        test_list = [t.eval_set.dataframe for t in train_test]
        for train, test in zip(train_list, test_list):
            assert not train.empty and not test.empty


class TestSplitterFailures:

    @pytest.mark.parametrize("params", p["invalid_temporal_holdout_test_ratio"])
    def test_invalid_or_missing_params_temporal_holdout_test_ratio(self, params):
        config = {
            "test_splitting": {
                "strategy": "temporal_holdout",
                **params
            }
        }

        with pytest.raises((ValueError, AttributeError)):
            load_and_split_data(config)

    @pytest.mark.parametrize("params", p["invalid_temporal_holdout_leave_n_out"])
    def test_invalid_or_missing_params_temporal_holdout_leave_n_out(self, params):
        config = {
            "test_splitting": {
                "strategy": "temporal_holdout",
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
                "strategy": "random_subsampling",
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
                "strategy": "random_subsampling",
                **params
            }
        }

        with pytest.raises((ValueError, AttributeError)):
            load_and_split_data(config)

    @pytest.mark.parametrize("params", p["invalid_random_cross_validation"])
    def test_invalid_or_missing_params_random_cross_validation(self, params):
        config = {
            "test_splitting": {
                "strategy": "random_cross_validation",
                **params
            }
        }

        with pytest.raises(ValueError):
            load_and_split_data(config)

    @pytest.mark.parametrize("params", p["invalid_fixed_timestamp"])
    def test_invalid_or_missing_params_fixed_timestamp(self, params):
        config = {
            "test_splitting": {
                "strategy": "fixed_timestamp",
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
                "strategy": "fixed_timestamp",
                **params
            }
        }

        with pytest.raises(ValueError):
            load_and_split_data(config)

    @pytest.mark.parametrize("params", p["invalid_strategy"])
    def test_invalid_or_missing_strategy(self, params):
        config = {
            "test_splitting": {
                **({"strategy": params["strategy"]} if params["strategy"] is not None else {}),
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
                "strategy": "fixed_timestamp",
                "timestamp": 8
            }
        }

        with pytest.raises(ValueError):
            load_and_split_data(config)


if __name__ == '__main__':
    pytest.main()
