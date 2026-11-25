from elliot.utils.folder import path_joiner
from tests import test_path, generate_param_combinations


hyperopt_data_folder = path_joiner(test_path, "hyperoptimization", "{0}")
hyperopt_dataset_path = lambda file_name=None: path_joiner(hyperopt_data_folder, f"{file_name or "dataset"}.tsv")


# Hyperoptimization

params_hyperoptimization_fail = {
    "grid_with_distributions": generate_param_combinations(
        ["neighborhood"],
        {"neighborhood": [["uniform", 10, 50], ["quniform", 10, 50, 10], ["randint", 50]]}
    ),
    "missing_max_evals_with_distributions": generate_param_combinations(
        [("hyper_opt_alg", "neighborhood")],
        {
            "hyper_opt_alg": ["tpe", "rand", "anneal", "mix"],
            "neighborhood": [["uniform", 10, 50], ["randint", 50]]
        }
    ),
    "invalid_max_evals": generate_param_combinations(
        [("hyper_opt_alg", "hyper_max_evals")],
        {
            "hyper_opt_alg": ["tpe", "grid"],
            "hyper_max_evals": [0, -3]
        }
    ),
    # The grid has 4 configurations
    "insufficient_grid_max_evals": generate_param_combinations(
        ["hyper_max_evals"],
        {"hyper_max_evals": [1, 3]}
    ),
    "invalid_opt_alg": generate_param_combinations(
        ["hyper_opt_alg"],
        {"hyper_opt_alg": ["invalid", 3, None]}
    ),
    # Values not valid for the type of the hyperparameter
    "invalid_param_values": generate_param_combinations(
        ["neighborhood"],
        {"neighborhood": [12.5, "invalid", None]}
    )
}
