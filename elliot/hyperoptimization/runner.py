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


@dataclass(frozen=True)
class RunOutcome:
    best_eval: dict
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
    data_test: List[DataSet],
    config: ExperimentConfig,
    model_config: RecommenderConfig,
    model_name: str,
    test_fold_index: int
) -> RunOutcome:
    if not requires_hyperopt(model_config):
        return run_single(
            data_test=data_test,
            config=config,
            model_config=model_config,
            model_name=model_name,
            test_fold_index=test_fold_index,
        )

    rstate = HyperoptRandomState(seed=config.random_seed)
    engine = HyperOptEngine(rstate=rstate)

    model_config.prepare_fields_for_search()

    coordinator = ModelCoordinator(data_test, config, model_config, model_name, test_fold_index)

    tuning = engine.optimize(
        coordinator=coordinator,
        model_config=model_config
    )

    params = tuning.best_params if tuning.best_trial is not None else None
    best_eval = coordinator.evaluate(params)

    return RunOutcome(
        best_eval=best_eval,
        trials=tuning.trials,
        all_trial_results=tuning.trials.results,
    )


def run_single(
    data_test: List[DataSet],
    config: ExperimentConfig,
    model_config: RecommenderConfig,
    model_name: str,
    test_fold_index: int
) -> RunOutcome:
    coordinator = ModelCoordinator(data_test, config, model_config, model_name, test_fold_index)
    best_eval = coordinator.single()

    return RunOutcome(
        best_eval=best_eval,
        trials=None,
        all_trial_results=[],
    )
