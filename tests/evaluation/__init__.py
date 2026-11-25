from elliot.utils.folder import path_joiner
from tests import test_path, generate_param_combinations


eval_data_folder = path_joiner(test_path, "evaluation", "{0}")
eval_dataset_path = lambda file_name=None: path_joiner(eval_data_folder, f"{file_name or "dataset"}.tsv")
user_filter_path = lambda file_name: path_joiner(eval_data_folder, "user_filter", f"{file_name}.tsv")


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
