"""
Facade for running hyperparameter optimization and final evaluation.
"""

from typing import List, Optional
from dataclasses import dataclass
import numpy as np

from elliot.dataset import DataSet
from elliot.hyperoptimization.engine import HyperOptEngine
from elliot.hyperoptimization.model_coordinator import ModelCoordinator
from elliot.namespace import RecommenderConfig, ExperimentConfig
from elliot.recommender import AbstractRecommender
from elliot.utils import get_trainer
from elliot.utils.registry import model_registry


@dataclass(frozen=True)
class RunOutcome:
    best_model: AbstractRecommender
    best_params: Optional[dict]
    results: dict
    trials: Optional[object]
    all_trial_results: list


class HyperoptRandomState:
    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed=seed)

    def integers(self, *args, **kwargs):
        value = self._rng.integers(*args, **kwargs)
        if isinstance(value, np.integer):
            return int(value)
        return value

    def __getattr__(self, name):
        return getattr(self._rng, name)


def requires_hyperopt(model_config: RecommenderConfig) -> bool:
    excluded = {"meta", "early_stopping", "best_iteration", "name"}
    for field_name in model_config.model_fields:
        if field_name in excluded:
            continue
        if isinstance(getattr(model_config, field_name), list):
            return True
    return False


def run_hyperopt(
    val_data: List[DataSet],
    main_data: DataSet,
    config: ExperimentConfig,
    model_config: RecommenderConfig,
    model_name: str,
    test_fold_index: int
) -> RunOutcome:

    rstate = HyperoptRandomState(seed=config.random_seed)
    engine = HyperOptEngine(rstate=rstate)

    model_config.prepare_fields_for_search()

    coordinator = ModelCoordinator(
        val_data,
        main_data,
        config,
        model_config,
        model_name,
        test_fold_index
    )

    tuning = engine.optimize(
        coordinator=coordinator,
        model_config=model_config
    )

    results = tuning.best_trial["result"] if tuning.best_trial is not None else {}
    best_params = tuning.best_params if tuning.best_trial is not None else None

    best_model = tuning.best_model
    if best_model is None:
        best_model = coordinator.train(best_params)

    return RunOutcome(
        best_model=best_model,
        best_params=best_params,
        results=results,
        trials=tuning.trials,
        all_trial_results=tuning.trials.results,
    )


def run_single(
    val_data: List[DataSet],
    main_data: DataSet,
    config: ExperimentConfig,
    model_config: RecommenderConfig,
    model_name: str,
    test_fold_index: int
) -> RunOutcome:

    coordinator = ModelCoordinator(
        val_data,
        main_data,
        config,
        model_config,
        model_name,
        test_fold_index
    )

    results = coordinator.single()
    best_params = results["params"]

    best_model = results.pop("best_model", None)
    if best_model is None:
        best_model = coordinator.train(best_params)

    return RunOutcome(
        best_model=best_model,
        best_params=best_params,
        results=results,
        trials=None,
        all_trial_results=[]
    )


def run_proxy(model_config, main_data, config):
    model = model_registry.get(
        name="ProxyRecommender",
        params=model_config,
        interactions=main_data.train_set,
        seed=config.random_seed
    )

    params = model_config.model_dump()

    results = {
        "name": model.name,
        "params": params,
        "loss": None
    }

    return RunOutcome(
        best_model=model,
        best_params=params,
        results=results,
        trials=None,
        all_trial_results=[]
    )


def run_evaluation(main_data, config, model):
    trainer = get_trainer(model)(
        config=config,
        model=model
    )

    return trainer.evaluate(main_data)
