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

    engine = HyperOptEngine(rstate=np.random.default_rng(config.random_seed))

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

    if tuning.best_trial is None:
        raise RuntimeError(
            f"Every hyperparameter trial of {model_name} failed: no configuration to train."
        )

    results = tuning.best_trial["result"]
    best_params = tuning.best_params

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
