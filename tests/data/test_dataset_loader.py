import importlib
from pathlib import Path
from tests.utils import *

import pytest
from unittest.mock import patch

current_path = Path(__file__).resolve().parent
data_path = Path.cwd().parent.parent / 'data'


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

    @pytest.mark.parametrize(
        "params",
        [
            {
                "folder_path": str(current_path / "fixed_strategy_with_validation"),
                "train_shape": 40,
                "val_shape": 5,
                "test_shape": 5,
            },
            {
                "folder_path": str(current_path / "fixed_strategy"),
                "train_shape": 45,
                "test_shape": 5,
            },
        ],
    )
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

    @pytest.mark.parametrize(
        "params", [
            {
                'root_folder': str(current_path / 'hierarchy_strategy'),
                "train_shapes": [[40, 40], 45],
                "val_shapes": [[5, 5]],
                "test_shapes": [5, 5],
            }
        ]
    )
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

    @pytest.mark.parametrize(
        "params", [
            {
                'dataset_folder': str(current_path / 'dataset_strategy'),
                'test_ratio': 0.2,
                'df_shape': 50
            },
            #{
            #    'dataset_folder': str(data_path / 'cat_dbpedia_movielens_1m_v030'),
            #    'test_ratio': 0.2,
            #    'df_shape': 1000209
            #}
            #{
            #    'dataset_folder': str(data_path / 'cat_dbpedia_movielens_10m'),
            #    'test_ratio': 0.2,
            #    'df_shape': 10000054
            #}
        ]
    )
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


class TestDataSetLoaderFailures:

    @pytest.mark.parametrize('val', [True, False])
    def test_fixed_strategy_missing_train_path(self, val):
        folder_path = current_path / (
            'fixed_strategy' if not val else 'fixed_strategy_with_validation'
        )

        config = {
            'strategy': 'fixed',
            'test_path': str(current_path / 'test.tsv'),
            **({'validation_path': str(folder_path / 'val.tsv')} if val else {})
        }

        with pytest.raises(AttributeError):
            dataloader(config)

    @pytest.mark.parametrize('val', [True, False])
    def test_fixed_strategy_missing_file(self, val):
        folder_path = current_path / 'fixed_strategy_missing_file'

        config = {
            'strategy': 'fixed',
            'train_path': str(folder_path / 'train.tsv'),
            'test_path': str(folder_path / 'test.tsv'),
            **({'validation_path': str(folder_path / 'val.tsv')} if val else {})
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

    def test_dataset_strategy_invalid_split(self, monkeypatch):
        dataset_path = str(current_path / 'dataset_strategy/dataset.tsv')

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


def apply_cleaning(df, users, items):
    cleaner_class = getattr(importlib.import_module("elliot.dataset"), 'Cleaner')
    cleaner = cleaner_class(df, users, items)
    return cleaner.process_cleaning()


class TestCleaner:
    @pytest.mark.parametrize(
        "params", [
            {
                'dataset_folder': str(current_path / 'filter_nan'),
                'users': [1],
                'items': [1],
                'df_final_shape': 2
            }
        ]
    )
    def test_filter_nan(self, params):
        dataset_path = params['dataset_folder'] + '/dataset.tsv'
        df_mock = read_dataset(dataset_path)

        cleaned_df = apply_cleaning(df_mock, set(params['users']), set(params['items']))

        assert cleaned_df.shape[0] == params['df_final_shape']
        assert cleaned_df.duplicated().sum() == 0
        assert not cleaned_df["timestamp"].isna().any()


if __name__ == '__main__':
    pytest.main()
