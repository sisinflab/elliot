# import functools
# import time
from itertools import product

from elliot.utils.folder import parent_dir, path_joiner

test_path = parent_dir(__file__)
data_folder = path_joiner(test_path, "data", "{0}")
dataset_path = path_joiner(data_folder, "dataset.tsv")


# def time_single_test(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         start = time.perf_counter()
#         try:
#             return func(*args, **kwargs)
#         finally:
#             end = time.perf_counter()
#             duration = end - start
#             print(f"[{func.__name__}] executed in {duration:.4f} seconds")
#     return wrapper


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
    "invalid_fixed": generate_param_combinations(
        ["data_folder"],
        {"data_folder": ["non/existent/path", 3, None]}
    ),
    "invalid_dataset": generate_param_combinations(
        ["dataset_path"],
        {"dataset_path": ["non/existent/path", [3], None]}
    ),
    "invalid_strategy": generate_param_combinations(
        ["strategy"],
        {"strategy": ["invalid", 3, None]}
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
    ),
    "invalid_strategy": generate_param_combinations(
        ["strategy"],
        {"strategy": ["invalid", 3, None]}
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


# NegativeSampler

params_neg_sampling_fail = {
    "invalid_neg_random": generate_param_combinations(
        [("num_negatives", "leave_one_out")],
        {
            "num_negatives": [20, [3], -5],
            "leave_one_out": [True, "invalid"]
        }
    ),
    "invalid_neg_fixed": generate_param_combinations(
        [("read_folder", "leave_one_out")],
        {
            "read_folder": ["./{0}", "non/existent/path", 3, None],
            "leave_one_out": [True, "invalid"]
        }
    ),
    "invalid_strategy": generate_param_combinations(
        ["strategy"],
        {"strategy": ["invalid", 3, None]}
    )
}


# Early stopping

params_early_stopping_fail = {
    "invalid": generate_param_combinations(
        [("monitor", "patience", "mode", "min_delta", "rel_delta", "baseline", "verbose")],
        {
            "monitor": ["loss", -3],
            "patience": [3, "invalid"],
            "mode": ["min", "invalid", [3]],
            "min_delta": [0.01, -3, "invalid"],
            "rel_delta": [0.05, -3, [3]],
            "baseline": [0.04, -3, "invalid"],
            "verbose": [True, "invalid"]
        }
    )
}
