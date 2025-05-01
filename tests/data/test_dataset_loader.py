import importlib
import sys
from pathlib import Path
import pandas as pd

import pytest
from unittest.mock import patch

current_path = Path(__file__).resolve().parent
sys.path.append(str(current_path))

from elliot.namespace.namespace_model_builder import NameSpaceBuilder


def dataloader(config_dict):
    base_folder_path_elliot = str(current_path.parents[1] / 'elliot')
    base_folder_path_config = str(current_path.parents[1] / 'config_files')

    def wrap_data_config(data_config):
        dataset = True if data_config['strategy'] == 'dataset' else False
        return {
            'experiment': {
                'dataset': 'test-dataset',
                'data_config': data_config,
                **({'splitting': {'test_splitting': {'strategy': 'random_subsampling'}}} if dataset else {})
            }
        }

    name_space = NameSpaceBuilder(
        base_folder_path_elliot, base_folder_path_config, config_dict=wrap_data_config(config_dict)
    )
    config = name_space.base.base_namespace
    dataloader_class = getattr(importlib.import_module("elliot.dataset"), config.data_config.dataloader)
    return dataloader_class(config)


def read_and_split_mock_dataset(dataset_path):
    column_names = ['userId', 'itemId', 'rating', 'timestamp']
    df_mock = pd.read_csv(dataset_path, sep='\t', names=column_names)
    train_df = df_mock.iloc[:4]
    test_df = df_mock.iloc[4:]
    return df_mock, train_df, test_df


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

        df_mock, train_df, test_df = read_and_split_mock_dataset(dataset_path)

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
