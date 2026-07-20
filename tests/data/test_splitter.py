import pytest

from elliot.dataset import DataSetLoader
from elliot.namespace import build_namespace
from elliot.utils.folder import path_joiner, check_path, parent_dir

from tests.params import params_splitting_fail as p, dataset_path

current_path = path_joiner(__file__)


def load_and_split_data(config_dict, file_name=None, seq=False, session_strategy=None):
    config_data = {
        "experiment": {
            "dataset": "splitter",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path(file_name),
                "sequential": seq,
                **({"session_strategy": session_strategy} if session_strategy is not None else {}),
                "reader": {"header": True}
            },
            "splitting": config_dict
        }
    }
    config = build_namespace(config_path=current_path, config_data=config_data)
    loader = DataSetLoader(config=config)
    val_data, main_data = loader.build()
    return val_data, main_data

def load_and_split_sequence_data(config_dict):
    return load_and_split_data(config_dict, "sequence", True)


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
        assert len(test) == 3
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

    def test_random_holdout_test_ratio(self):
        config = {
            "test_splitting": {
                "strategy": "random_holdout",
                "test_ratio": 0.1
            }
        }

        _, train_test = load_and_split_data(config)

        assert len(train_test) == 1
        train = train_test[0].train_set.dataframe
        test = train_test[0].eval_set.dataframe
        assert not train.empty and not test.empty
        assert len(test) == 3
        assert len(train) + len(test) == 30

    def test_random_holdout_leave_n_out(self):
        config = {
            "test_splitting": {
                "strategy": "random_holdout",
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
                "timestamp": 21600
            }
        }

        _, train_test = load_and_split_data(config)

        assert len(train_test) == 1
        train = train_test[0].train_set.dataframe
        test = train_test[0].eval_set.dataframe
        assert not train.empty and not test.empty
        assert all(test["timestamp"] >= 21600)
        assert all(train["timestamp"] < 21600)

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
        save_folder = "./splitter/splitting"
        config = {
            "save_on_disk": True,
            "save_folder": save_folder,
            "test_splitting": {
                "strategy": "fixed_timestamp",
                "timestamp": 25200
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


class TestSequenceSplitter:

    def test_temporal_holdout_leave_n_out(self):
        config = {
            "test_splitting": {
                "strategy": "temporal_holdout",
                "leave_n_out": 1
            }
        }

        _, train_test = load_and_split_sequence_data(config)

        assert len(train_test) == 1
        train = train_test[0].train_set.dataframe
        test = train_test[0].eval_set.dataframe
        assert len(train) + len(test) == 15
        assert list(test["itemId"]) == ["4", "5", "7", "8", "6", "7"]

    def test_dropping_users_with_a_single_session(self):
        config = {
            "test_splitting": {
                "strategy": "temporal_holdout",
                "leave_n_out": 1
            }
        }

        _, train_test = load_and_split_data(config, "dropping_users", session_strategy="session_only")

        train = train_test[0].train_set.dataframe
        test = train_test[0].eval_set.dataframe

        assert "2" not in set(train["userId"]) | set(test["userId"])
        assert list(test["itemId"]) == ["4", "5", "6"]
        assert list(train["itemId"]) == ["1", "2", "3"]

    def test_flat_strategy_keeps_single_session_users(self):
        """With the default FLAT session strategy no segmentation happens at all,
        so there is no notion of a "single session" user to drop: every user survives."""
        config = {
            "test_splitting": {
                "strategy": "temporal_holdout",
                "leave_n_out": 1
            }
        }

        _, train_test = load_and_split_data(config, "dropping_users")

        train = train_test[0].train_set.dataframe
        test = train_test[0].eval_set.dataframe

        assert set(train["userId"]) | set(test["userId"]) == {"1", "2"}

    def test_never_splitting_sessions(self):
        config = {
            "test_splitting": {
                "strategy": "random_cross_validation",
                "folds": 2
            }
        }

        _, train_test = load_and_split_sequence_data(config)

        for fold in train_test:
            train = fold.train_set.dataframe
            test = fold.eval_set.dataframe
            train_sessions = set(zip(train["userId"], train["sessionId"]))
            test_sessions = set(zip(test["userId"], test["sessionId"]))
            assert not (train_sessions & test_sessions)


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

    @pytest.mark.parametrize("params", p["invalid_random_holdout_test_ratio"])
    def test_invalid_or_missing_params_random_holdout_test_ratio(self, params):
        config = {
            "test_splitting": {
                "strategy": "random_holdout",
                **params
            }
        }

        with pytest.raises((ValueError, AttributeError)):
            load_and_split_data(config)

    @pytest.mark.parametrize("params", p["invalid_random_holdout_leave_n_out"])
    def test_invalid_or_missing_params_random_holdout_leave_n_out(self, params):
        config = {
            "test_splitting": {
                "strategy": "random_holdout",
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

    def test_fixed_timestamp_rejected(self):
        config = {
            "test_splitting": {
                "strategy": "fixed_timestamp",
                "timestamp": 2
            }
        }

        with pytest.raises(ValueError):
            load_and_split_sequence_data(config)


if __name__ == '__main__':
    pytest.main()
