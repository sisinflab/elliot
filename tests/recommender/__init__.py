from tests import generate_param_combinations


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
