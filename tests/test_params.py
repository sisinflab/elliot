from tests.test_utils import test_path, data_path

_folder_movielens_1m = str(data_path / 'cat_dbpedia_movielens_1m_v030')
_path_movielens_1m = _folder_movielens_1m + '/dataset.tsv'

_folder_movielens_10m = str(data_path / 'cat_dbpedia_movielens_10m')
_path_movielens_10m = _folder_movielens_10m + '/dataset.tsv'


params_dataset_loader = {
    'fixed_strategy': [
        {
            "folder_path": str(test_path / "fixed_strategy_with_validation"),
            "train_shape": 40,
            "val_shape": 5,
            "test_shape": 5,
        },
        {
            "folder_path": str(test_path / "fixed_strategy"),
            "train_shape": 45,
            "test_shape": 5,
        }
    ],
    'hierarchy_strategy': [
        {
            'root_folder': str(test_path / 'hierarchy_strategy'),
            "train_shapes": [[40, 40], 45],
            "val_shapes": [[5, 5]],
            "test_shapes": [5, 5],
        }
    ],
    'dataset_strategy': [
        {
            'dataset_folder': str(test_path / 'dataset_strategy'),
            'test_ratio': 0.2,
            'df_shape': 50
        },
        #{
        #    'dataset_folder': _folder_movielens_1m,
        #    'test_ratio': 0.2,
        #    'df_shape': 1000209
        #},
        #{
        #    'dataset_folder': _folder_movielens_10m,
        #    'test_ratio': 0.2,
        #    'df_shape': 10000054
        #}
    ],
    'fixed_strategy_missing_file': [
        {
            'folder_path': str(test_path / 'fixed_strategy_missing_file')
        }
    ],
    'filter_nan': [
        {
            'dataset_folder': str(test_path / 'filter_nan'),
            'df_final_shape': 2
        }
    ]
}


params_pre_filtering = {
    #'dataset_path': _path_movielens_10m,
    'global_threshold': [
        {
            'dataset_path': str(test_path / 'filter_ratings_by_global_threshold/filter_ratings_by_global_threshold.tsv'),
            'threshold': 3
        }
    ],
    'user_average': [
        {
            'dataset_path': str(test_path / 'filter_ratings_by_user_average/filter_ratings_by_user_average.tsv'),
        }
    ],
    'user_k_core': [
        {
            'dataset_path': str(test_path / 'filter_user_k_core/filter_user_k_core.tsv'),
            'core': 3
        }
    ],
    'item_k_core': [
        {
            'dataset_path': str(test_path / 'filter_item_k_core/filter_item_k_core.tsv'),
            'core': 3
        }
    ],
    'iterative_k_core': [
        {
            'dataset_path': str(test_path / 'filter_iterative_k_core/filter_iterative_k_core.tsv'),
            'core': 2
        }
    ],
    'n_rounds_k_core': [
        {
            'dataset_path': str(test_path / 'filter_n_rounds_k_core/filter_n_rounds_k_core.tsv'),
            'core': 2,
            'rounds': 2
        }
    ],
    'cold_users': [
        {
            'dataset_path': str(test_path / 'filter_retain_cold_users/filter_retain_cold_users.tsv'),
            'threshold': 2
        }
    ]
}
