from itertools import product

from elliot.utils.enums import DataLoadingStrategy

from tests.utils import data_path


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

params_dataset_loader_fail = {
    "invalid_or_missing_params": generate_param_combinations(
        [("strategy", "data_path", "header")],
        {
            "strategy": [DataLoadingStrategy.DATASET.value, 'invalid', 3, None],
            "data_path": [data_path, 3, None],
            "header": [False, 3]
        }
    )
}


# PreFilter

params_pre_filtering_fail = {
    "invalid_global_threshold": generate_param_combinations(
        ["threshold"],
        {"threshold": [-3, "invalid"]}
    ),
    "invalid_user_k_core": generate_param_combinations(
        ["core"],
        {"core": [-5, [3]]}
    ),
    "invalid_item_k_core": generate_param_combinations(
        ["core"],
        {"core": [-5, 2.5]}
    ),
    "invalid_iterative_k_core": generate_param_combinations(
        ["core"],
        {"core": [-5, "invalid"]}
    ),
    "invalid_n_rounds_combinations": generate_param_combinations(
        [("core", "rounds")],
        {
            "core": [2, -5, [3]],
            "rounds": [2, -5, [3]]
        }
    ),
    "invalid_cold_users": generate_param_combinations(
        ["threshold"],
        {"threshold": [-5, "invalid", None]}
    )
}


# Splitter

params_splitting_fail = {
    "invalid_temporal_holdout_test_ratio": generate_param_combinations(
        ["test_ratio"],
        {"test_ratio": [0.0, 2.5, [3], None]}
    ),
    "invalid_temporal_holdout_leave_n_out": generate_param_combinations(
        ["leave_n_out"],
        {"leave_n_out": [300, -3, "invalid", None]}
    ),
    "invalid_random_subsampling_test_ratio": generate_param_combinations(
        [("folds", "test_ratio")],
        {
            "folds": [3, 31, [3], None],
            "test_ratio": [0.1, 1.0, "invalid", None]
        }
    ),
    "invalid_random_subsampling_leave_n_out": generate_param_combinations(
        [("folds", "leave_n_out")],
        {
            "folds": [3, 31, "invalid", None],
            "leave_n_out": [2, 200, 2.5, None]
        }
    ),
    "invalid_random_cross_validation": generate_param_combinations(
        ["folds"],
        {"folds": [31, 2.5]}
    ),
    "invalid_fixed_timestamp": generate_param_combinations(
        ["timestamp"],
        {
            "timestamp": [50, [3]]
        }
    ),
    "invalid_best_timestamp": generate_param_combinations(
        [("min_below", "min_over")],
        {
            "min_below": [1, 100, "invalid"],
            "min_over": [1, 100, "invalid"]
        }
    ),
    "invalid_strategy": generate_param_combinations(
        ["strategy"],
        {"strategy": ["invalid", 3, None]}
    )
}
