# import functools
# import time
from itertools import product

from elliot.utils.folder import parent_dir, path_joiner

test_path = parent_dir(__file__)
data_folder = path_joiner(test_path, "data", "{0}")
dataset_path = lambda file_name=None: path_joiner(data_folder, f"{file_name or "dataset"}.tsv")

eval_data_folder = path_joiner(test_path, "evaluation", "{0}")
eval_dataset_path = lambda file_name=None: path_joiner(eval_data_folder, f"{file_name or "dataset"}.tsv")
user_filter_path = lambda file_name: path_joiner(eval_data_folder, "user_filter", f"{file_name}.tsv")

config_path = path_joiner(test_path, "configs", "test.yml")

reader_folder = path_joiner(test_path, "utils", "reader")
reader_path = lambda file_name=None, ext="tsv", folder=None: (
    path_joiner(reader_folder, folder or f"{file_name}.{ext}")
)

writer_folder = path_joiner(test_path, "utils", "writer")
writer_path = lambda file_name=None, ext="tsv", folder=None: (
    path_joiner(writer_folder, folder or f"{file_name}.{ext}")
)


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
        [("data_folder", "remove_duplicates", "sequential", "session_strategy")],
        {
            "data_folder": ["non/existent/path", 3, None],
            "remove_duplicates": [True, "invalid"],
            "sequential": [False, "invalid"],
            "session_strategy": ["flat", "invalid"]
        }
    ),
    "invalid_dataset": generate_param_combinations(
        [("dataset_path", "remove_duplicates", "sequential", "session_strategy")],
        {
            "dataset_path": ["non/existent/path", [3], None],
            "remove_duplicates": [True, "invalid"],
            "sequential": [False, "invalid"],
            "session_strategy": ["flat", "invalid"]
        }
    ),
    "invalid_strategy": generate_param_combinations(
        [("strategy", "remove_duplicates", "sequential", "session_strategy")],
        {
            "strategy": ["invalid", 3, None],
            "remove_duplicates": [True, "invalid"],
            "sequential": [False, "invalid"],
            "session_strategy": ["flat", "invalid"]
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
    "invalid_random_holdout_test_ratio": generate_param_combinations(
        ["test_ratio"],
        {"test_ratio": [0.0, 2.5, [3], None]}
    ),
    "invalid_random_holdout_leave_n_out": generate_param_combinations(
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
    "invalid_fixed_timestamp_test_ratio": generate_param_combinations(
        ["test_ratio"],
        {
            "test_ratio": [1.0, "invalid"]
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


# EvalProtocol

params_eval_protocol_fail = {
    "invalid_neg_random": generate_param_combinations(
        ["num_negatives"],
        {
            "num_negatives": [[3], -5]
        }
    ),
    "invalid_neg_fixed": generate_param_combinations(
        ["read_folder"],
        {
            "read_folder": ["non/existent/path", 3, None]
        }
    ),
    "invalid_strategy": generate_param_combinations(
        ["strategy"],
        {"strategy": ["invalid", 3, None]}
    ),
    "missing_user_filter_file": generate_param_combinations(
        ["path"],
        {"path": [user_filter_path("missing"), "non/existent/path"]}
    ),
    "invalid_user_filter_id_space": generate_param_combinations(
        ["id_space"],
        {"id_space": ["internal", 3]}
    ),
    "invalid_session_only_target": generate_param_combinations(
        ["target"],
        {"target": ["all", "first"]}
    )
}


# Namespace

params_namespace_fail = {
    "invalid_target_for_mixed_session_strategies": generate_param_combinations(
        ["target"],
        {"target": ["all", "first", "random", "successive"]}
    ),
    "invalid_splitting_for_mixed_session_strategies": generate_param_combinations(
        ["test_splitting"],
        {
            "test_splitting": [
                {"strategy": "temporal_holdout", "leave_n_out": 2},
                {"strategy": "temporal_holdout", "test_ratio": 0.2},
                {"strategy": "fixed_timestamp", "timestamp": 1000}
            ]
        }
    ),
    # Prefix-based targets need a temporal splitting
    "invalid_splitting_for_evaluation_target": generate_param_combinations(
        ["splitting"],
        {
            "splitting": [
                {"test_splitting": {"strategy": "random_holdout", "test_ratio": 0.2}},
                {
                    "test_splitting": {"strategy": "temporal_holdout", "test_ratio": 0.2},
                    "validation_splitting": {"strategy": "random_cross_validation", "folds": 2}
                }
            ]
        }
    ),
    "invalid_evaluation_target_for_session_models": generate_param_combinations(
        ["target"],
        {"target": ["all", "first"]}
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
