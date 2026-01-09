import pytest
from pathlib import Path

from elliot.dataset import DataSetLoader
from elliot.utils.enums import DataLoadingStrategy

from tests.params import params_dataset_loader_fail as p
from tests.utils import create_namespace, data_path

current_path = Path(__file__).parent


def load_data(config_dict):
    data_config = {
        "experiment": {**config_dict}
    }
    ns = create_namespace(data_config, current_path)
    loader = DataSetLoader(ns)
    return loader.dataframe


class TestDataSetLoader:

    def test_fixed_strategy(self):
        config = {
            "dataset": "fixed_strategy",
            "data_config": {
                "strategy": DataLoadingStrategy.FIXED.value,
                "data_path": data_path
            }
        }

        df = load_data(config)

        assert df[0][1].shape[0] == 5
        assert df[0][0].shape[0] == 45

    def test_fixed_strategy_with_validation(self):
        config = {
            "dataset": "fixed_strategy_with_validation",
            "data_config": {
                "strategy": DataLoadingStrategy.FIXED.value,
                "data_path": data_path
            }
        }

        df = load_data(config)

        assert df[0][1].shape[0] == 5
        assert df[0][0][0][0].shape[0] == 40
        assert df[0][0][0][1].shape[0] == 5
        assert len(df[0][0]) == 1

    def test_hierarchy_strategy(self):
        config = {
            "dataset": "hierarchy_strategy",
            "data_config": {
                "strategy": DataLoadingStrategy.HIERARCHY.value,
                "data_path": data_path
            }
        }

        df = load_data(config)

        assert df[0][1].shape[0] == 5
        assert df[0][0][0][0].shape[0] == 40
        assert df[0][0][0][1].shape[0] == 5
        assert df[0][0][1][0].shape[0] == 40
        assert df[0][0][1][1].shape[0] == 5

        assert df[1][1].shape[0] == 5
        assert df[1][0].shape[0] == 45

    def test_dataset_strategy(self):
        config = {
            "dataset": "dataset_strategy",
            "data_config": {
                "strategy": DataLoadingStrategy.DATASET.value,
                "data_path": data_path
            }
        }

        df = load_data(config)

        assert df.shape[0] == 50

    def test_filter_nan(self):
        config = {
            "dataset": "filter_nan",
            "data_config": {
                "strategy": DataLoadingStrategy.DATASET.value,
                "data_path": data_path
            }
        }

        df = load_data(config)

        assert df.shape[0] == 2
        assert df.duplicated().sum() == 0
        assert not df["timestamp"].isna().any()


class TestDataSetLoaderFailures:

    @pytest.mark.parametrize("params", p["invalid_or_missing_params"])
    def test_invalid_or_missing_params(self, params):
        if (
            params.get("strategy") == DataLoadingStrategy.DATASET.value and
            params.get("data_path") == data_path and
            params.get("header") == False
        ):
            pytest.skip("Test requires at least one invalid parameter to be meaningful.")

        config = {
            "dataset": "dataset_strategy",
            "data_config": {
                **({"strategy": params["strategy"]} if params["strategy"] is not None else {}),
                **({"data_path": params["data_path"]} if params["data_path"] is not None else {}),
                "header": params["header"]
            }
        }

        with pytest.raises(ValueError):
            load_data(config)

    def test_missing_folder(self):
        config = {
            "dataset": "dataset_strategy",
            "data_config": {
                "strategy": DataLoadingStrategy.DATASET.value,
                "data_path": "non/existent/path"
            }
        }

        with pytest.raises(FileNotFoundError):
            load_data(config)

    def test_missing_file(self):
        config = {
            "dataset": "missing_file",
            "data_config": {
                "strategy": DataLoadingStrategy.DATASET.value,
                "data_path": data_path
            }
        }

        with pytest.raises(FileNotFoundError):
            load_data(config)


if __name__ == '__main__':
    pytest.main()
