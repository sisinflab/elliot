import pytest

from elliot.dataset import DataSetLoader
from elliot.namespace import build_namespace
from elliot.utils.folder import path_joiner

from tests.params import params_dataset_loader_fail as p
from tests.utils import data_folder, dataset_path

current_path = path_joiner(__file__)


def load_data(config_dict):
    config_data = {
        "experiment": {**config_dict}
    }
    config = build_namespace(config_path=current_path, config_data=config_data)
    dataset_loader = DataSetLoader(config=config)
    return dataset_loader.interactions_df


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

        assert df[0][1].shape[0] == 5
        assert df[0][0][0][0].shape[0] == 45

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

        assert df[0][1].shape[0] == 5
        assert df[0][0][0][0].shape[0] == 40
        assert df[0][0][0][1].shape[0] == 5
        assert len(df[0][0]) == 1

    def test_hierarchy(self):
        config = {
            "dataset": "hierarchy_strategy",
            "data_config": {
                "strategy": "hierarchy",
                "data_folder": data_folder
            }
        }

        df = load_data(config)

        assert df[0][1].shape[0] == 5
        assert df[0][0][0][0].shape[0] == 40
        assert df[0][0][0][1].shape[0] == 5
        assert df[0][0][1][0].shape[0] == 40
        assert df[0][0][1][1].shape[0] == 5

        assert df[1][1].shape[0] == 5
        assert df[1][0][0][0].shape[0] == 45

    def test_dataset(self):
        config = {
            "dataset": "dataset_strategy",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path,
                "reader": {"header": True}
            }
        }

        df = load_data(config)

        assert df.shape[0] == 50

    def test_filter_nan(self):
        config = {
            "dataset": "filter_nan",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path,
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
                "dataset_path": dataset_path
            }
        }

        with pytest.raises(ValueError):
            load_data(config)

    def test_missing_required_column(self):
        config = {
            "dataset": "missing_required_column",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path,
                "reader": {"header": True}
            }
        }

        with pytest.raises(KeyError):
            load_data(config)


if __name__ == '__main__':
    pytest.main()
