from itertools import product
from tests.utils import test_path, data_path

_folder_movielens_1m = str(data_path / 'movielens_1m_v030')
_path_movielens_1m = _folder_movielens_1m + '/dataset.tsv'

_folder_movielens_10m = str(data_path / 'movielens_10m')
_path_movielens_10m = _folder_movielens_10m + '/dataset.tsv'

_folder_movielens_20m = str(data_path / 'movielens_20m')
_path_movielens_20m = _folder_movielens_20m + '/dataset.tsv'


def generate_param_combinations(key_list, values, base=None):
    if base is None:
        base_list = [{}] * len(key_list)
    elif not isinstance(base, list):
        base_list = [base] * len(key_list)
    else:
        base_list = base
    result = []
    for keys, base in zip(key_list, base_list):
        if not isinstance(keys, tuple):
            keys = (keys,)
        value_lists = [values[k] for k in keys]
        for combo in product(*value_lists):
            overrides = dict(zip(keys, combo))
            config = {**base, **overrides}
            result.append(config)
    return result


# DataSetLoader

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
        # movielens_1m
        #{
        #    'dataset_folder': _folder_movielens_1m,
        #    'test_ratio': 0.2,
        #    'df_shape': 1000209
        #},
        # movielens_10m
        #{
        #    'dataset_folder': _folder_movielens_10m,
        #    'test_ratio': 0.2,
        #    'df_shape': 10000054
        #}
    ],
    'filter_nan': [
        {
            'dataset_folder': str(test_path / 'filter_nan'),
            'df_final_shape': 2
        }
    ]
}

params_dataset_loader_fail = {
    'fixed_strategy_missing_file': [
        {
            'folder_path': str(test_path / 'fixed_strategy_missing_file')
        }
    ],
    'hierarchy_strategy_missing_root_folder': [
        {
            'root_folder': 'non/existent/path'
        }
    ],
    'dataset_strategy_missing_dataset': [
        {
            'dataset_path': 'nonexistent/file.tsv'
        }
    ]
}


# PreFilter

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
            'core': 2
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

params_pre_filtering_fail = {
    'invalid_global_threshold': generate_param_combinations(
        ['threshold'],
        {'threshold': [[3], -3, 'invalid']},
        params_pre_filtering['global_threshold']
    ),
    'invalid_user_k_core': generate_param_combinations(
        ['core'],
        {'core': [[3], -5, 'abc']},
        params_pre_filtering['user_k_core']
    ),
    'invalid_item_k_core': generate_param_combinations(
        ['core'],
        {'core': [[3], -5, 2.5]},
        params_pre_filtering['item_k_core']
    ),
    'invalid_iterative_k_core': generate_param_combinations(
        ['core'],
        {'core': [[3], -5, 'x']},
        params_pre_filtering['iterative_k_core']
    ),
    'invalid_n_rounds_combinations': generate_param_combinations(
        [('core', 'rounds')],
        {
            'core': [2, [3], -5, 'x'],
            'rounds': [2, [3], -5, 'y']
        },
        params_pre_filtering['n_rounds_k_core']
    ),
    'invalid_cold_users': generate_param_combinations(
        ['threshold'],
        {'threshold': [-99, 'cold', None]},
        params_pre_filtering['global_threshold']
    )
}


# Splitter

params_splitting = {
    'dataset_path': str(test_path / 'splitting_strategies/dataset.tsv'),
    #'dataset_path': _path_movielens_20m,
    'save_folder': str(test_path / 'splitting_strategies/splitting'),
    'temporal_holdout': [
        {
            'test_ratio': 0.1,
        },
        {
            'leave_n_out': 3
        },
        # movielens_1m
        # {
        #    'leave_n_out': 17
        # },
        # movielens_10m, movielens_20m
        # {
        #    'leave_n_out': 15
        # }
    ],
    'random_subsampling': [
        {
            'folds': 10,
            'test_ratio': 0.1
        },
        {
            'folds': 3,
            'leave_n_out': 2
        },
        # movielens_1m, movielens_10m, movielens_20m
        # {
        #    'folds': 10,
        #    'leave_n_out': 17
        # }
    ],
    'random_cross_validation': [
        {
            'folds': 10
        }
    ],
    'fixed_timestamp': [
        {
            'timestamp': 7
        },
        {
            'min_below': 1,
            'min_over': 1
        },
        # movielens_1m
        # {
        #    'timestamp': 974687965
        # },
        # movielens_10m
        # {
        #    'timestamp': 1079842786
        # },
        # movielens_20m
        # {
        #     'timestamp': 1139534337
        # },
    ]
}

params_splitting_fail = {
    'invalid_temporal_holdout': generate_param_combinations(
        ['test_ratio', 'leave_n_out'],
        {
            'test_ratio': [0.0, [3], -3, 'x', None],
            'leave_n_out': [300, 2.5, -3, 'y', None]
        }
    ),
    'invalid_random_subsampling': generate_param_combinations(
        [('folds', 'test_ratio'), ('folds', 'leave_n_out')],
        {
            'folds': [3, 31, 2.5, -3, 'abc', None],
            'test_ratio': [0.1, 1.0, [3], -3, 'invalid', None],
            'leave_n_out': [2, 200, 2.5, -3, 'z', None]
        }
    ),
    'invalid_random_cross_validation': generate_param_combinations(
        ['folds'],
        {'folds': [31, 2.5, -3, 'fold']}
    ),
    'invalid_fixed_timestamp': generate_param_combinations(
        ['timestamp', ('min_below', 'min_over')],
        {
            'timestamp': [50, [3], -3, 'time'],
            'min_below': [1, 100, 2.5, -3, 'below'],
            'min_over': [1, 100, 2.5, -3, 'over']
        }
    )
}
