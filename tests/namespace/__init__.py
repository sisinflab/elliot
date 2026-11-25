from elliot.utils.folder import path_joiner
from tests import test_path, generate_param_combinations


config_path = path_joiner(test_path, "configs", "test.yml")


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
