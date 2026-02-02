"""
Facade for running hyperparameter optimization and final evaluation.
"""

from dataclasses import dataclass
from typing import Optional

from elliot.hyperoptimization.engine import HyperOptEngine
from elliot.hyperoptimization.model_coordinator import ModelCoordinator
from elliot.hyperoptimization.policy import FinalPolicy


@dataclass(frozen=True)
class RunOutcome:
    best_eval: dict
    trials: Optional[object]
    all_trial_results: list


class HyperoptRunner:
    def __init__(self, engine: HyperOptEngine):
        self.engine = engine
        self._final_policy = FinalPolicy()

    def run(self, data_test, base_namespace, model_base, model_class, test_fold_index: int) -> RunOutcome:
        coordinator = ModelCoordinator(data_test, base_namespace, model_base, model_class, test_fold_index)
        if isinstance(model_base, tuple):
            tuning = self.engine.optimize(
                coordinator=coordinator,
                space=model_base[1],
                algo=model_base[3],
                max_evals=model_base[2],
            )
            if tuning.best_trial is None:
                best_eval = coordinator.run(self._final_policy)
            else:
                best_eval = coordinator.run(self._final_policy, tuning.best_params)
            return RunOutcome(
                best_eval=best_eval,
                trials=tuning.trials,
                all_trial_results=tuning.trials.results,
            )

        best_eval = coordinator.run(self._final_policy)
        return RunOutcome(best_eval=best_eval, trials=None, all_trial_results=[best_eval])
