from elliot.utils.folder import path_joiner
from tests import test_path, generate_param_combinations


data_folder = path_joiner(test_path, "data", "{0}")
dataset_path = lambda file_name=None: path_joiner(data_folder, f"{file_name or "dataset"}.tsv")


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
