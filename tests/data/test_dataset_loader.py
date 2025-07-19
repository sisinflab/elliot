import importlib
from tests.utils import *
from tests.params import params_dataset_loader

import pytest
from unittest.mock import patch


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

    @pytest.mark.parametrize('params', params_dataset_loader['fixed_strategy'])
    @time_single_test
    def test_fixed_strategy(self, params):
        val = True if 'val_shape' in params.keys() else False

        config = {
            'strategy': 'fixed',
            'train_path': params['folder_path'] + '/train.tsv',
            'test_path': params['folder_path'] + '/test.tsv',
            **({'validation_path': params['folder_path'] + '/val.tsv'} if val else {})
        }

        loader = dataloader(config)

        def check_dataloader(data, shape):
            data.shuffle = False
            assert data.shape[0] == shape

        check_dataloader(loader.tuple_list[0][1], params['test_shape'])
        if val:
            check_dataloader(loader.tuple_list[0][0][0][0], params['train_shape'])
            check_dataloader(loader.tuple_list[0][0][0][1], params['val_shape'])
            assert len(loader.tuple_list[0][0]) == 1
        else:
            check_dataloader(loader.tuple_list[0][0], params['train_shape'])

    @pytest.mark.parametrize('params', params_dataset_loader['hierarchy_strategy'])
    def test_hierarchy_strategy(self, params):
        config = {
            'strategy': 'hierarchy',
            'root_folder': params['root_folder']
        }

        loader = dataloader(config)

        i = 0
        for t in loader.tuple_list:
            assert t[1].shape[0] == params['test_shapes'][i]
            if isinstance(t[0], list):
                j = 0
                for train_val in t[0]:
                    assert train_val[0].shape[0] == params['train_shapes'][i][j]
                    assert train_val[1].shape[0] == params['val_shapes'][i][j]
                    j += 1
            else:
                assert t[0].shape[0] == params['train_shapes'][i]
            i += 1

    @pytest.mark.parametrize('params', params_dataset_loader['dataset_strategy'])
    @time_single_test
    def test_dataset_strategy(self, params):
        dataset_path = params['dataset_folder'] + '/dataset.tsv'
        config = {
            'strategy': 'dataset',
            'dataset_path': dataset_path
        }

        df_mock = read_dataset(dataset_path)
        split_idx = round(df_mock.shape[0] * (1 - params['test_ratio']))
        train_df = df_mock.iloc[:split_idx]
        test_df = df_mock.iloc[split_idx:]

        with (
            patch(
                "elliot.dataset.loader_coordinator.Splitter.process_splitting",
                return_value=[(train_df, test_df)]
            ) as mock_splitter
        ):
            loader = dataloader(config)

            mock_splitter.assert_called_once()

            assert loader.dataframe.shape[0] == params['df_shape']
            assert loader.tuple_list[0][0].equals(train_df)
            assert loader.tuple_list[0][1].equals(test_df)

    @pytest.mark.parametrize('params', params_dataset_loader['filter_nan'])
    def test_filter_nan(self, params):
        dataset_path = params['dataset_folder'] + '/dataset.tsv'
        config = {
            'strategy': 'dataset',
            'dataset_path': dataset_path
        }

        df_mock = read_dataset(dataset_path)

        with (
            patch(
                "elliot.dataset.loader_coordinator.Splitter.process_splitting",
                return_value=[(df_mock, df_mock)]
            )
        ):
            loader = dataloader(config)

            assert loader.dataframe.shape[0] == params['df_final_shape']
            assert loader.dataframe.duplicated().sum() == 0
            assert not loader.dataframe["timestamp"].isna().any()


class TestDataSetLoaderFailures:

    @pytest.mark.parametrize('params', params_dataset_loader['fixed_strategy'])
    def test_fixed_strategy_missing_train_path(self, params):
        val = True if 'val_shape' in params.keys() else False

        config = {
            'strategy': 'fixed',
            'test_path': params['folder_path'] + '/test.tsv',
            **({'validation_path': params['folder_path'] + '/val.tsv'} if val else {})
        }

        with pytest.raises(AttributeError):
            dataloader(config)

    @pytest.mark.parametrize('params', params_dataset_loader['fixed_strategy_missing_file'])
    def test_fixed_strategy_missing_file(self, params):
        val = True if 'val_shape' in params.keys() else False

        config = {
            'strategy': 'fixed',
            'train_path': params['folder_path'] + '/train.tsv',
            'test_path': params['folder_path'] + '/test.tsv',
            **({'validation_path': params['folder_path'] + '/val.tsv'} if val else {})
        }

        with pytest.raises(FileNotFoundError):
            dataloader(config)

    def test_hierarchy_strategy_missing_root_folder(self):
        config = {
            'strategy': 'hierarchy',
            'root_folder': 'non/existent/path'
        }

        with pytest.raises(FileNotFoundError):
            dataloader(config)

    def test_dataset_strategy_missing_dataset(self):
        config = {
            'strategy': 'dataset',
            'dataset_path': 'nonexistent/file.tsv'
        }

        with pytest.raises(FileNotFoundError):
            dataloader(config)

    @pytest.mark.parametrize('params', params_dataset_loader['dataset_strategy'])
    def test_dataset_strategy_invalid_split(self, params, monkeypatch):
        dataset_path = params['dataset_folder'] + '/dataset.tsv'

        def fake_splitter(*args, **kwargs):
            return [] # Failed split

        monkeypatch.setattr(
            "elliot.dataset.loader_coordinator.Splitter.process_splitting",
            fake_splitter
        )

        config = {
            'strategy': 'dataset',
            'dataset_path': dataset_path
        }

        with pytest.raises(IndexError):
            dataloader(config)


if __name__ == '__main__':
    pytest.main()
