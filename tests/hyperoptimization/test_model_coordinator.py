import pytest
from hyperopt import STATUS_OK

from elliot.dataset import DataSetLoader
from elliot.evaluation.protocol import prepare_protocols
from elliot.hyperoptimization import ModelCoordinator
from elliot.namespace import build_namespace
from elliot.utils.folder import path_joiner

from tests.hyperoptimization import params_hyperoptimization_fail as p, hyperopt_dataset_path

current_path = path_joiner(__file__)

no_validation = {"test_splitting": {"strategy": "temporal_holdout", "test_ratio": 0.2}}
with_validation = {**no_validation, "validation_splitting": {"strategy": "temporal_holdout", "test_ratio": 0.2}}


def get_coordinator(model_name, model_dict, splitting=no_validation):
    config_data = {
        "experiment": {
            "dataset": "hyperopt_pipeline",
            "random_seed": 42,
            "data_config": {
                "strategy": "dataset",
                "dataset_path": hyperopt_dataset_path(),
                "reader": {"header": True}
            },
            "splitting": splitting,
            "evaluation": {"cutoffs": [5], "simple_metrics": ["nDCG"]},
            "top_k": 5,
            "models": {model_name: model_dict}
        }
    }
    config = build_namespace(config_path=current_path, config_data=config_data)
    val_data, main_data = DataSetLoader(config=config).build()
    prepare_protocols(config, val_data, main_data)
    return ModelCoordinator(val_data[0], main_data[0], config, config.models[model_name], model_name, 0)


class TestModelCoordinator:

    def test_objective(self):
        coordinator = get_coordinator("ItemKNN", {"neighborhood": [3, 5], "similarity": ["cosine", "jaccard"]})

        payload = coordinator.objective({"neighborhood": 3, "similarity": "jaccard"})

        assert payload["status"] == STATUS_OK
        assert payload["params"]["neighborhood"] == 3
        assert payload["params"]["similarity"] == "jaccard"
        # The validation metric is maximized: its opposite is the loss
        assert payload["objective"]["target"] == "validation_metric"
        assert payload["objective"]["direction"] == "maximize"
        assert payload["val_metric"] == payload["val_results"][5]["nDCG"]
        assert payload["loss"] == -payload["val_metric"]
        assert payload["trend"]["val_metric_name"] == "nDCG@5"
        # The explored values are applied to a copy: the search space is left as it is
        assert coordinator.model_config.neighborhood == [3, 5]

    def test_objective_on_internal_loss(self):
        coordinator = get_coordinator("BPRMF", {
            "meta": {"optimize_internal_loss": True},
            "epochs": 2,
            "factors": [4, 8]
        })

        payload = coordinator.objective({"factors": 8})

        assert payload["objective"]["target"] == "internal_loss"
        assert payload["objective"]["direction"] == "minimize"
        assert payload["loss"] == min(payload["trend"]["train_loss"])
        assert payload["trend"]["epochs"] == [1, 2]

    @pytest.mark.parametrize("metric,expected_loss", [("nDCG", -0.3), ("RMSE", 0.3)])
    def test_objective_direction(self, metric, expected_loss):
        coordinator = get_coordinator("ItemKNN", {"neighborhood": [3, 5]})
        coordinator.model_config.meta.validation_metric = f"{metric}@5"

        objective = coordinator._compute_objective({5: {metric: 0.3}}, [], coordinator.model_config)

        assert objective["loss"] == expected_loss

    def test_quantized_values_are_cast(self):
        coordinator = get_coordinator("BPRMF", {"factors": ["quniform", 4, 16, 4]})

        # q-distributions sample floats, even for int hyperparameters
        payload = coordinator.objective({"factors": 8.0})

        assert payload["params"]["factors"] == 8
        assert isinstance(payload["params"]["factors"], int)

    @pytest.mark.parametrize("splitting,reused", [(no_validation, True), (with_validation, False)])
    def test_best_model_reuse(self, splitting, reused):
        coordinator = get_coordinator("ItemKNN", {"neighborhood": [3, 5]}, splitting)

        payload = coordinator.objective({"neighborhood": 3})

        # The model is reused for testing only when trained on the test fold's own train
        assert ("best_model" in payload) == reused

    def test_single(self):
        coordinator = get_coordinator("ItemKNN", {"neighborhood": 3})

        payload = coordinator.single()

        assert payload["status"] == STATUS_OK
        assert payload["params"]["neighborhood"] == 3
        assert "trend" not in payload

    def test_train(self):
        coordinator = get_coordinator("ItemKNN", {"neighborhood": [3, 5]}, with_validation)

        model = coordinator.train({"neighborhood": 5})

        assert model.neighborhood == 5


class TestModelCoordinatorFailures:

    @pytest.mark.parametrize("params", p["invalid_param_values"])
    def test_invalid_param_values(self, params):
        coordinator = get_coordinator("ItemKNN", {"neighborhood": [3, 5]})

        with pytest.raises(ValueError):
            coordinator.objective({"neighborhood": params["neighborhood"]})


if __name__ == "__main__":
    pytest.main()
