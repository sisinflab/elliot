import pytest

from elliot.dataset import DataSetLoader
from elliot.evaluation.protocol import prepare_protocols
from elliot.hyperoptimization import (
    ModelCoordinator, run_hyperopt, run_single, run_proxy, run_evaluation, requires_hyperopt
)
from elliot.namespace import build_namespace
from elliot.utils.folder import path_joiner

from tests.hyperoptimization import hyperopt_dataset_path

current_path = path_joiner(__file__)


def get_run_args(model_name, model_dict):
    config_data = {
        "experiment": {
            "dataset": "hyperopt_pipeline",
            "random_seed": 42,
            "data_config": {
                "strategy": "dataset",
                "dataset_path": hyperopt_dataset_path(),
                "reader": {"header": True}
            },
            "splitting": {
                "test_splitting": {"strategy": "temporal_holdout", "test_ratio": 0.2},
                "validation_splitting": {"strategy": "temporal_holdout", "test_ratio": 0.2}
            },
            "evaluation": {"cutoffs": [5], "simple_metrics": ["nDCG"]},
            "top_k": 5,
            "models": {model_name: model_dict}
        }
    }
    config = build_namespace(config_path=current_path, config_data=config_data)
    val_data, main_data = DataSetLoader(config=config).build()
    prepare_protocols(config, val_data, main_data)
    return {
        "val_data": val_data[0],
        "main_data": main_data[0],
        "config": config,
        "model_config": config.models[model_name],
        "model_name": model_name,
        "test_fold_index": 0
    }


class TestRunner:

    @pytest.mark.parametrize("model_dict,expected", [
        ({"neighborhood": [3, 5]}, True),
        ({"neighborhood": ["quniform", 3, 9, 3]}, True),
        ({"neighborhood": 3}, False),
        # Unknown fields are not hyperparameters of the model
        ({"neighborhood": 3, "neighbors": [3, 5]}, False)
    ])
    def test_requires_hyperopt(self, model_dict, expected):
        args = get_run_args("ItemKNN", model_dict)

        assert requires_hyperopt(args["model_config"]) == expected

    def test_run_hyperopt_grid(self):
        args = get_run_args("ItemKNN", {
            "meta": {"hyper_opt_alg": "grid"},
            "neighborhood": [3, 5],
            "similarity": ["cosine", "jaccard"]
        })

        outcome = run_hyperopt(**args)

        assert len(outcome.all_trial_results) == 4
        assert all("best_model" not in result for result in outcome.all_trial_results)
        assert outcome.results["loss"] == min(outcome.trials.losses())
        assert outcome.results["params"]["neighborhood"] == outcome.best_params["neighborhood"]
        # Validation folds train on a subset of the test train: the best model is retrained
        assert outcome.best_model.neighborhood == outcome.best_params["neighborhood"]
        assert outcome.best_model.similarity == outcome.best_params["similarity"]

    def test_run_hyperopt_search(self):
        args = get_run_args("BPRMF", {
            "meta": {"hyper_opt_alg": "tpe", "hyper_max_evals": 3},
            "factors": ["quniform", 4, 16, 4],
            "learning_rate": ["loguniform", -6, -2]
        })

        outcome = run_hyperopt(**args)

        assert len(outcome.all_trial_results) == 3
        assert outcome.results["loss"] == min(outcome.trials.losses())
        # Quantized samples reach the model with the type of the hyperparameter
        assert isinstance(outcome.best_model.factors, int)
        assert outcome.best_model.factors == outcome.best_params["factors"]

    def test_run_single(self):
        args = get_run_args("ItemKNN", {"neighborhood": 3})

        outcome = run_single(**args)

        assert outcome.trials is None
        assert outcome.all_trial_results == []
        assert outcome.best_params["neighborhood"] == 3
        assert outcome.best_model.neighborhood == 3

    def test_run_proxy(self, tmp_path):
        recs_path = path_joiner(tmp_path, "recs.tsv")
        with open(recs_path, "w") as file:
            file.write("1\t1\t0.9\n1\t2\t0.8\n")
        args = get_run_args("ProxyRecommender", {"path": recs_path})

        outcome = run_proxy(args["model_config"], args["main_data"], args["config"])

        assert outcome.trials is None
        assert outcome.best_params["path"] == recs_path
        assert outcome.results["name"] == "recs"
        assert outcome.results["loss"] is None

    def test_run_evaluation(self):
        args = get_run_args("ItemKNN", {"neighborhood": 3})
        outcome = run_single(**args)

        results = run_evaluation(args["main_data"], args["config"], outcome.best_model)

        assert 0 <= results["test_results"][5]["nDCG"] <= 1


class TestRunnerFailures:

    def test_all_trials_failed(self, monkeypatch):
        def failing_objective(self, params):
            raise RuntimeError("Trial failed")

        monkeypatch.setattr(ModelCoordinator, "objective", failing_objective)
        args = get_run_args("ItemKNN", {
            "meta": {"hyper_opt_alg": "grid"},
            "neighborhood": [3, 5]
        })

        with pytest.raises(RuntimeError, match="Every hyperparameter trial"):
            run_hyperopt(**args)


if __name__ == "__main__":
    pytest.main()
