import importlib
from pathlib import Path
from tests.utils import *

import pytest
from unittest.mock import patch

current_path = Path(__file__).resolve().parent


def dataloader(config_dict):
    def wrap_data_config(data_config):
        data_config['side_information'] = []
        return {
            'data_config': data_config,
            'splitting': {'test_splitting': {'strategy': 'random_subsampling'}},
            'random_seed': 42,
            'binarize': False,
            'config_test': False
        }
    ns = create_namespace(wrap_data_config(config_dict))
    dataloader_class = getattr(importlib.import_module("elliot.dataset"), 'DataSetLoader')
    return dataloader_class(ns)


class TestDataSetLoader:

    @pytest.mark.parametrize('val', [True, False])
    def test_fixed_strategy(self, val):
        folder_path = current_path / (
            'fixed_strategy_with_validation' if val else 'fixed_strategy'
        )

        config = {
            'strategy': 'fixed',
            'train_path': str(folder_path / 'train.tsv'),
            'test_path': str(folder_path / 'test.tsv'),
            **({'validation_path': str(folder_path / 'val.tsv')} if val else {})
        }

        loader = dataloader(config)

        assert loader.train_dataframe.shape[0] == 1
        assert loader.test_dataframe.shape[0] == 1
        if val:
            assert loader.validation_dataframe.shape[0] == 1
        else:
            assert hasattr(loader, "validation_dataframe") is False
        assert isinstance(loader.tuple_list, list)
        assert len(loader.tuple_list[0][0]) == 1

    def test_hierarchy_strategy(self):
        root_folder = str(current_path / 'hierarchy_strategy')
        config = {
            'strategy': 'hierarchy',
            'root_folder': root_folder
        }

        loader = dataloader(config)

        assert isinstance(loader.tuple_list, list)
        assert len(loader.tuple_list[0][0]) == 2
        assert len(loader.tuple_list[1]) == 2

    def test_dataset_strategy(self):
        dataset_path = str(current_path / 'dataset_strategy/dataset.tsv')
        config = {
            'strategy': 'dataset',
            'dataset_path': dataset_path
        }

        df_mock = read_dataset(dataset_path)
        train_df = df_mock.iloc[:4]
        test_df = df_mock.iloc[4:]

        with (
            patch(
                "elliot.dataset.loader_coordinator.PreFilter.filter",
                return_value=df_mock
            ) as mock_filter,
            patch(
                "elliot.dataset.loader_coordinator.Splitter.process_splitting",
                return_value=[(train_df, test_df)]
            ) as mock_splitter
        ):

            loader = dataloader(config)

            mock_filter.assert_called_once()
            mock_splitter.assert_called_once()

            assert loader.dataframe.shape[0] == 5
            assert loader.tuple_list[0][0].equals(train_df)
            assert loader.tuple_list[0][1].equals(test_df)


if __name__ == '__main__':
    pytest.main()
