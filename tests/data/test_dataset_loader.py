import pytest

from elliot.dataset import DataSetLoader
from elliot.namespace import build_namespace
from elliot.utils.enums import SessionStrategy
from elliot.utils.folder import path_joiner

from tests.params import params_dataset_loader_fail as p, data_folder, dataset_path

current_path = path_joiner(__file__)


def get_loader(config_dict):
    config_data = {
        "experiment": config_dict
    }
    config = build_namespace(config_path=current_path, config_data=config_data)
    return DataSetLoader(config=config)

def load_data(config_dict):
    loader = get_loader(config_dict)
    return loader.dataframe


class TestDataSetLoader:

    def test_fixed(self):
        config = {
            "dataset": "fixed_strategy",
            "data_config": {
                "strategy": "fixed",
                "data_folder": data_folder,
                "reader": {"header": True}
            }
        }

        df = load_data(config)

        folds, train, test = df[0]
        assert folds == []
        assert train.shape[0] == 45
        assert test.shape[0] == 5

    def test_fixed_with_validation(self):
        config = {
            "dataset": "fixed_strategy_with_validation",
            "data_config": {
                "strategy": "fixed",
                "data_folder": data_folder,
                "reader": {"header": True}
            }
        }

        df = load_data(config)

        folds, train, test = df[0]
        assert folds[0][0].shape[0] == 40
        assert folds[0][1].shape[0] == 5
        assert len(df[0][0]) == 1
        assert train is None
        assert test.shape[0] == 5

    def test_hierarchy(self):
        config = {
            "dataset": "hierarchy_strategy",
            "data_config": {
                "strategy": "hierarchy",
                "data_folder": data_folder
            }
        }

        df = load_data(config)

        folds, train, test = df[0]
        assert folds[0][0].shape[0] == 40
        assert folds[0][1].shape[0] == 5
        assert folds[1][0].shape[0] == 40
        assert folds[1][1].shape[0] == 5
        assert train.shape[0] == 45
        assert test.shape[0] == 5

        folds, train, test = df[1]
        assert folds == []
        assert train.shape[0] == 45
        assert test.shape[0] == 5

    def test_dataset(self):
        config = {
            "dataset": "dataset_strategy",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path(),
                "reader": {"header": True}
            }
        }

        df = load_data(config)

        assert df.shape[0] == 50

    def test_session_strategy_defaults_to_flat(self):
        config = {
            "dataset": "dataset_strategy",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path(),
                "reader": {"header": True}
            }
        }

        loader = get_loader(config)

        assert loader.data_config.session_strategy == SessionStrategy.FLAT

    def test_session_strategy_explicit_session_only(self):
        config = {
            "dataset": "dataset_strategy",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path(),
                "session_strategy": "session_only",
                "reader": {"header": True}
            }
        }

        loader = get_loader(config)

        assert loader.data_config.session_strategy == SessionStrategy.SESSION_ONLY

    def test_fixed_sequential(self):
        config = {
            "dataset": "fixed_strategy_sequential",
            "data_config": {
                "strategy": "fixed",
                "data_folder": data_folder,
                "sequential": True,
                "reader": {"format": "inline"}
            }
        }

        df = load_data(config)

        folds, train, test = df[0]
        assert folds == []
        assert train.shape[0] == 45
        assert test.shape[0] == 5

    def test_fixed_with_validation_sequential(self):
        config = {
            "dataset": "fixed_strategy_with_validation_sequential",
            "data_config": {
                "strategy": "fixed",
                "data_folder": data_folder,
                "sequential": True,
                "reader": {"format": "inline"}
            }
        }

        df = load_data(config)

        folds, train, test = df[0]
        assert folds[0][0].shape[0] == 40
        assert folds[0][1].shape[0] == 5
        assert len(df[0][0]) == 1
        assert train is None
        assert test.shape[0] == 5

    def test_hierarchy_sequential(self):
        config = {
            "dataset": "hierarchy_strategy_sequential",
            "data_config": {
                "strategy": "hierarchy",
                "data_folder": data_folder,
                "sequential": True,
                "reader": {"format": "inline"}
            }
        }

        df = load_data(config)

        folds, train, test = df[0]
        assert folds[0][0].shape[0] == 40
        assert folds[0][1].shape[0] == 5
        assert folds[1][0].shape[0] == 40
        assert folds[1][1].shape[0] == 5
        assert train.shape[0] == 45
        assert test.shape[0] == 5

        folds, train, test = df[1]
        assert folds == []
        assert train.shape[0] == 45
        assert test.shape[0] == 5

    def test_dataset_sequential(self):
        config = {
            "dataset": "dataset_strategy_sequential",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path(),
                "sequential": True,
                "reader": {"format": "inline"}
            }
        }

        df = load_data(config)

        assert df.shape[0] == 50

    def test_filter_nan(self):
        config = {
            "dataset": "dataset_loader",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path("filter_nan"),
                "reader": {"header": True}
            }
        }

        df = load_data(config)

        assert df.shape[0] == 2
        assert df.duplicated().sum() == 0
        assert not df["timestamp"].isna().any()


class TestDataSetLoaderFailures:

    @pytest.mark.parametrize("params", p["invalid_fixed"])
    def test_invalid_or_missing_params_fixed(self, params):
        config = {
            "dataset": "fixed_strategy",
            "data_config": {
                "strategy": "fixed",
                **({"data_folder": params["data_folder"]} if params["data_folder"] is not None else {}),
                "sequential": params["sequential"]
            }
        }

        with pytest.raises((FileNotFoundError, ValueError, AttributeError)):
            load_data(config)

    @pytest.mark.parametrize("params", p["invalid_dataset"])
    def test_invalid_or_missing_params_dataset(self, params):
        config = {
            "dataset": "fixed_strategy",
            "data_config": {
                "strategy": "dataset",
                **({"dataset_path": params["dataset_path"]} if params["dataset_path"] is not None else {}),
                "sequential": params["sequential"]
            }
        }

        with pytest.raises((FileNotFoundError, ValueError, AttributeError)):
            load_data(config)

    @pytest.mark.parametrize("params", p["invalid_strategy"])
    def test_invalid_or_missing_strategy(self, params):
        config = {
            "dataset": "dataset_strategy",
            "data_config": {
                **({"strategy": params["strategy"]} if params["strategy"] is not None else {}),
                "dataset_path": dataset_path(),
                "sequential": params["sequential"]
            }
        }

        with pytest.raises(ValueError):
            load_data(config)

    def test_missing_required_column(self):
        config = {
            "dataset": "dataset_loader",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path("missing_required_column"),
                "reader": {"header": True}
            }
        }

        with pytest.raises(KeyError):
            load_data(config)


class TestSequenceProcessing:

    def test_sequential_forces_session_only(self):
        config = {
            "dataset": "dataset_loader",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path("sequence_wide"),
                "sequential": True,
                # Explicitly requesting FLAT must still be overridden, since
                # sequential source rows are already organized in sessions.
                "session_strategy": "flat",
                "reader": {"header": False}
            }
        }

        loader = get_loader(config)

        assert loader.data_config.session_strategy == SessionStrategy.SESSION_ONLY

    def test_wide_synthesizes_order_key(self):
        config = {
            "dataset": "dataset_loader",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path("sequence_wide"),
                "sequential": True,
                "reader": {"header": False}
            }
        }

        loader = get_loader(config)
        df = loader.dataframe

        assert loader.has_real_timestamps is False
        assert df.shape[0] == 12
        assert list(df.columns) == ["userId", "itemId", "sessionId", "timestamp", "rating"]
        assert (df["rating"] == 1.0).all()

        u1 = df[df["userId"] == "1"].sort_values("timestamp")
        assert list(u1["itemId"]) == ["1", "2", "3", "4", "5"]
        assert list(u1["timestamp"]) == [0, 1, 2, 3, 4]
        assert list(u1["sessionId"]) == [0, 0, 0, 0, 0]

    def test_inline_with_real_timestamp(self):
        config = {
            "dataset": "dataset_loader",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path("sequence_inline"),
                "sequential": True,
                "reader": {"format": "inline"}
            }
        }

        loader = get_loader(config)
        df = loader.dataframe

        assert loader.has_real_timestamps is True
        assert df.shape[0] == 12
        u1 = df[df["userId"] == "1"]
        u2 = df[df["userId"] == "2"]
        u3 = df[df["userId"] == "3"]
        assert not u1.empty and not u2.empty
        assert (u1["timestamp"] == 10).all()
        assert (u2["timestamp"] == 20).all()
        assert (u3["timestamp"] == 30).all()

    def test_inline_without_timestamp(self):
        config = {
            "dataset": "dataset_loader",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path("sequence_inline_no_timestamp"),
                "sequential": True,
                "reader": {"format": "inline"}
            }
        }

        loader = get_loader(config)
        df = loader.dataframe

        assert loader.has_real_timestamps is False
        u1 = df[df["userId"] == "1"].sort_values("timestamp")
        assert list(u1["itemId"]) == ["1", "2", "3", "4", "5"]
        assert list(u1["timestamp"]) == [0, 1, 2, 3, 4]

    def test_rating_always_implicit(self):
        config = {
            "dataset": "dataset_loader",
            "binarize": False,
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path("sequence_wide"),
                "sequential": True,
                "reader": {"header": False}
            }
        }

        df = load_data(config)

        assert (df["rating"] == 1.0).all()

    def test_multi_row_user(self):
        config = {
            "dataset": "dataset_loader",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path("multi_row_user"),
                "sequential": True,
                "reader": {"header": False}
            }
        }

        df = load_data(config)

        u1 = df[df["userId"] == "1"].sort_values("timestamp")
        assert list(u1["sessionId"]) == [0, 0, 0, 1, 1]

    def test_interactions_with_real_timestamp_segments_sessions(self):
        config = {
            "dataset": "dataset_loader",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path("session_gap"),
                "session_strategy": "session_only",
                "reader": {"header": True}
            }
        }

        loader = get_loader(config)
        df = loader.dataframe

        assert loader.has_real_timestamps is True

        u1 = df[df["userId"] == "1"].sort_values("timestamp")
        assert list(u1["sessionId"]) == [0, 0, 0, 1, 1, 1]

        u2 = df[df["userId"] == "2"]
        assert u2["sessionId"].nunique() == 1

    def test_interactions_with_real_timestamp_default_to_flat(self):
        config = {
            "dataset": "dataset_loader",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path("session_gap"),
                "reader": {"header": True}
            }
        }

        loader = get_loader(config)
        df = loader.dataframe

        assert loader.has_real_timestamps is True
        assert "sessionId" not in df.columns

    def test_interactions_without_timestamp_get_no_session_column(self):
        config = {
            "dataset": "dataset_loader",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path("no_timestamp_column"),
                "reader": {"header": True}
            }
        }

        loader = get_loader(config)
        df = loader.dataframe

        assert loader.has_real_timestamps is False
        assert "sessionId" not in df.columns
        assert "timestamp" in df.columns

    def test_fixed_with_real_timestamp_segments_sessions(self):
        config = {
            "dataset": "fixed_strategy_session_gap",
            "data_config": {
                "strategy": "fixed",
                "data_folder": data_folder,
                "session_strategy": "session_only",
                "reader": {"header": True}
            }
        }

        loader = get_loader(config)
        _, train, test = loader.dataframe[0]

        assert loader.has_real_timestamps is True

        u1_train = train[train["userId"] == "1"].sort_values("timestamp")
        assert list(u1_train["sessionId"]) == [0, 0, 0, 1, 1, 1]
        u2_train = train[train["userId"] == "2"]
        assert u2_train["sessionId"].nunique() == 1

        u1_test = test[test["userId"] == "1"]
        assert u1_test["sessionId"].nunique() == 1
        u2_test = test[test["userId"] == "2"].sort_values("timestamp")
        assert list(u2_test["sessionId"]) == [0, 0, 1]

    def test_hierarchy_with_real_timestamp_segments_sessions(self):
        config = {
            "dataset": "hierarchy_strategy_session_gap",
            "data_config": {
                "strategy": "hierarchy",
                "data_folder": data_folder,
                "session_strategy": "session_only",
                "reader": {"header": True}
            }
        }

        folds, train, test = load_data(config)[0]

        assert folds == []
        u1_train = train[train["userId"] == "1"].sort_values("timestamp")
        assert list(u1_train["sessionId"]) == [0, 0, 0, 1, 1, 1]
        u2_test = test[test["userId"] == "2"].sort_values("timestamp")
        assert list(u2_test["sessionId"]) == [0, 0, 1]

    def test_fixed_without_session_only_gets_no_session_column(self):
        config = {
            "dataset": "fixed_strategy_session_gap",
            "data_config": {
                "strategy": "fixed",
                "data_folder": data_folder,
                "reader": {"header": True}
            }
        }

        _, train, test = load_data(config)[0]

        assert "sessionId" not in train.columns
        assert "sessionId" not in test.columns


if __name__ == '__main__':
    pytest.main()
