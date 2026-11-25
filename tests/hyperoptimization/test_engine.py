import pytest
import logging
from itertools import product
import numpy as np
from hyperopt import STATUS_OK, STATUS_FAIL

from elliot.hyperoptimization import HyperOptEngine
from elliot.hyperoptimization.engine import build_hyperopt_space, OPT_ALGORITHMS
from elliot.namespace import build_namespace
from elliot.utils.enums import OptimizationAlgorithm
from elliot.utils.folder import path_joiner

from tests.hyperoptimization import params_hyperoptimization_fail as p, hyperopt_dataset_path

current_path = path_joiner(__file__)

search_algorithms = ["tpe", "rand", "anneal", "mix"]


def get_model_config(model_dict, model_name="ItemKNN"):
    config_data = {
        "experiment": {
            "dataset": "hyperopt_pipeline",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": hyperopt_dataset_path()
            },
            "evaluation": {"simple_metrics": ["nDCG"]},
            "top_k": 10,
            "models": {model_name: model_dict}
        }
    }
    config = build_namespace(config_path=current_path, config_data=config_data)
    model_config = config.models[model_name]
    model_config.prepare_fields_for_search()
    return model_config


class _FakeCoordinator:
    """Score each configuration as minus the sum of `key`'s value: the highest one is the best.
    Trials whose params match every item of `fail_on` raise (an empty dict fails them all)."""

    logger = logging.getLogger("fake-coordinator")

    def __init__(self, key="neighborhood", fail_on=None):
        self.key = key
        self.fail_on = fail_on
        self.explored = []

    def objective(self, params):
        self.explored.append(params)
        if self.fail_on is not None and all(params[k] == v for k, v in self.fail_on.items()):
            raise RuntimeError("Trial failed")
        return {
            "loss": -float(np.sum(params[self.key])),
            "status": STATUS_OK,
            "best_model": dict(params)
        }


class TestHyperOptEngine:

    def _optimize(self, model_config, coordinator):
        engine = HyperOptEngine(rstate=np.random.default_rng(42))
        return engine.optimize(coordinator=coordinator, model_config=model_config)

    def test_search_space_from_lists(self):
        model_config = get_model_config({
            "meta": {"hyper_opt_alg": "grid"},
            "neighborhood": [10, 20],
            "similarity": ["cosine", "jaccard"]
        })

        space, max_evals, algo = build_hyperopt_space(model_config)

        # Every declared hyperparameter is a choice, scalars included (a single value)
        assert {"neighborhood", "similarity", "eval_batch_size", "implicit"} <= set(space)
        assert all(node.name == "switch" for node in space.values())
        assert max_evals == 4
        assert algo == OptimizationAlgorithm.GRID.value

    def test_unknown_fields_are_not_searched(self):
        model_config = get_model_config({
            "neighborhood": 10,
            "neighbors": [10, 20]
        })

        space, max_evals, _ = build_hyperopt_space(model_config)

        assert "neighbors" not in space
        assert max_evals == 1

    @pytest.mark.parametrize("hyper_opt_alg", search_algorithms)
    def test_search_space_from_distributions(self, hyper_opt_alg):
        model_config = get_model_config({
            "meta": {"hyper_opt_alg": hyper_opt_alg, "hyper_max_evals": 5},
            "neighborhood": ["quniform", 10, 50, 10],
            "similarity": ["cosine", "jaccard"]
        })

        space, max_evals, algo = build_hyperopt_space(model_config)

        assert space["neighborhood"].name != "switch"
        assert space["similarity"].name == "switch"
        assert max_evals == 5
        assert algo is OPT_ALGORITHMS[OptimizationAlgorithm(hyper_opt_alg)]

    def test_max_evals_defaults_to_grid_size(self):
        model_config = get_model_config({
            "neighborhood": [10, 20, 30],
            "similarity": ["cosine", "jaccard"]
        })

        _, max_evals, _ = build_hyperopt_space(model_config)

        assert max_evals == 6

    @pytest.mark.parametrize("hyper_max_evals", [None, 4, 10])
    def test_grid_search(self, hyper_max_evals):
        model_config = get_model_config({
            "meta": {"hyper_opt_alg": "grid", "hyper_max_evals": hyper_max_evals},
            "neighborhood": [10, 20],
            "similarity": ["cosine", "jaccard"]
        })
        coordinator = _FakeCoordinator()

        tuning = self._optimize(model_config, coordinator)

        # Every combination exactly once, whatever `hyper_max_evals`
        explored = [(params["neighborhood"], params["similarity"]) for params in coordinator.explored]
        assert sorted(explored) == sorted(product([10, 20], ["cosine", "jaccard"]))
        assert len(tuning.trials) == 4

        # Scalars left to their default (`eval_batch_size` is None) are searched as they are
        assert all(params["eval_batch_size"] is None for params in coordinator.explored)

        assert tuning.best_params["neighborhood"] == 20
        assert tuning.best_loss == -20
        assert tuning.best_model == tuning.best_params
        assert all("best_model" not in result for result in tuning.trials.results)

    @pytest.mark.parametrize("mlp_hidden_size", [
        ["(16, 8)", "(32, 16)"],
        ["choice", "(16, 8)", "(32, 16)"]
    ])
    def test_grid_search_on_tuples(self, mlp_hidden_size):
        model_config = get_model_config({
            "meta": {"hyper_opt_alg": "grid"},
            "mlp_hidden_size": mlp_hidden_size
        }, model_name="NeuMF")
        coordinator = _FakeCoordinator(key="mlp_hidden_size")

        tuning = self._optimize(model_config, coordinator)

        assert [params["mlp_hidden_size"] for params in coordinator.explored] == [(16, 8), (32, 16)]
        assert tuning.best_params["mlp_hidden_size"] == (32, 16)

    @pytest.mark.parametrize("hyper_opt_alg", search_algorithms)
    def test_search(self, hyper_opt_alg):
        model_config = get_model_config({
            "meta": {"hyper_opt_alg": hyper_opt_alg, "hyper_max_evals": 6},
            "neighborhood": ["quniform", 10, 50, 10],
            "similarity": ["cosine", "jaccard"]
        })
        coordinator = _FakeCoordinator()

        tuning = self._optimize(model_config, coordinator)

        assert len(tuning.trials) == len(coordinator.explored) == 6
        assert tuning.best_loss == min(tuning.trials.losses())
        assert tuning.best_params["neighborhood"] == -tuning.best_loss
        assert 10 <= tuning.best_params["neighborhood"] <= 50
        assert tuning.best_params["similarity"] in ["cosine", "jaccard"]
        assert tuning.best_model["neighborhood"] == tuning.best_params["neighborhood"]

    @pytest.mark.parametrize("hyper_opt_alg", search_algorithms)
    def test_sampled_values_are_native(self, hyper_opt_alg):
        model_config = get_model_config({
            "meta": {"hyper_opt_alg": hyper_opt_alg, "hyper_max_evals": 4},
            "neighborhood": ["randint", 10, 50]
        })
        coordinator = _FakeCoordinator()

        tuning = self._optimize(model_config, coordinator)

        # hyperopt samples numpy scalars (`randint` gives `np.int64`): Python ones are handed over
        assert all(type(params["neighborhood"]) is int for params in coordinator.explored)
        assert type(tuning.best_params["neighborhood"]) is int

    @pytest.mark.parametrize("hyper_opt_alg", ["grid", "tpe"])
    def test_failed_trials_are_skipped(self, hyper_opt_alg):
        model_config = get_model_config({
            "meta": {"hyper_opt_alg": hyper_opt_alg, "hyper_max_evals": 8},
            "neighborhood": [10, 20],
            "similarity": ["cosine", "jaccard"]
        })
        # The highest neighborhood fails: the best trial is the lowest one
        coordinator = _FakeCoordinator(fail_on={"neighborhood": 20})

        tuning = self._optimize(model_config, coordinator)

        statuses = [result["status"] for result in tuning.trials.results]
        assert STATUS_FAIL in statuses
        assert STATUS_OK in statuses
        assert tuning.best_params["neighborhood"] == 10
        assert tuning.best_trial["result"]["status"] == STATUS_OK


class TestHyperOptEngineFailures:

    @pytest.mark.parametrize("params", p["grid_with_distributions"])
    def test_grid_with_distributions(self, params):
        model_config = get_model_config({
            "meta": {"hyper_opt_alg": "grid", "hyper_max_evals": 10},
            "neighborhood": params["neighborhood"]
        })

        with pytest.raises(ValueError):
            build_hyperopt_space(model_config)

    @pytest.mark.parametrize("params", p["missing_max_evals_with_distributions"])
    def test_missing_max_evals_with_distributions(self, params):
        model_config = get_model_config({
            "meta": {"hyper_opt_alg": params["hyper_opt_alg"]},
            "neighborhood": params["neighborhood"]
        })

        with pytest.raises(ValueError):
            build_hyperopt_space(model_config)

    @pytest.mark.parametrize("params", p["invalid_max_evals"])
    def test_invalid_max_evals(self, params):
        model_config = get_model_config({
            "meta": {"hyper_opt_alg": params["hyper_opt_alg"], "hyper_max_evals": params["hyper_max_evals"]},
            "neighborhood": [10, 20]
        })

        with pytest.raises(ValueError):
            build_hyperopt_space(model_config)

    @pytest.mark.parametrize("params", p["insufficient_grid_max_evals"])
    def test_insufficient_grid_max_evals(self, params):
        model_config = get_model_config({
            "meta": {"hyper_opt_alg": "grid", "hyper_max_evals": params["hyper_max_evals"]},
            "neighborhood": [10, 20],
            "similarity": ["cosine", "jaccard"]
        })

        with pytest.raises(ValueError):
            build_hyperopt_space(model_config)

    @pytest.mark.parametrize("params", p["invalid_opt_alg"])
    def test_invalid_opt_alg(self, params):
        with pytest.raises(ValueError):
            get_model_config({
                "meta": {"hyper_opt_alg": params["hyper_opt_alg"]},
                "neighborhood": [10, 20]
            })

    @pytest.mark.parametrize("hyper_opt_alg", ["grid", "tpe"])
    def test_all_trials_failed(self, hyper_opt_alg):
        model_config = get_model_config({
            "meta": {"hyper_opt_alg": hyper_opt_alg},
            "neighborhood": [10, 20]
        })
        engine = HyperOptEngine(rstate=np.random.default_rng(42))

        tuning = engine.optimize(coordinator=_FakeCoordinator(fail_on={}), model_config=model_config)

        # Every failure is recorded, and no trial is elected as the best one
        assert all(result["status"] == STATUS_FAIL for result in tuning.trials.results)
        assert tuning.best_trial is None
        assert tuning.best_model is None
        assert tuning.best_params == {}
        assert tuning.best_loss == np.inf


if __name__ == "__main__":
    pytest.main()
