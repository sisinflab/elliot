"""
Hyperopt optimization engine for Elliot.
"""

from __future__ import annotations

import copy
from functools import partial, reduce
from typing import Any, Dict, Optional
from dataclasses import dataclass
from collections import OrderedDict
from itertools import product

import numpy as np
from hyperopt import Trials, STATUS_FAIL, STATUS_OK, fmin, space_eval, hp, anneal, atpe, mix, rand, tpe
from hyperopt.base import JOB_STATE_DONE
from hyperopt.exceptions import AllTrialsFailed
from hyperopt.pyll import rec_eval

from elliot.hyperoptimization import patch_hyperopt, to_native
from elliot.namespace import RecommenderConfig
from elliot.recommender import AbstractRecommender
from elliot.utils.enums import OptimizationAlgorithm, SearchSpace

patch_hyperopt()


@dataclass(frozen=True)
class TuningResult:
    trials: Trials
    best_model: Optional[AbstractRecommender]
    best_params: Dict[str, Any]
    best_loss: float
    best_trial: Optional[dict]
    best_raw: Optional[dict]


class HyperOptEngine:
    def __init__(self, rstate: Optional[np.random.Generator] = None, show_progressbar: bool = False):
        self._rstate = rstate
        self._show_progressbar = show_progressbar

    def optimize(self, coordinator, model_config) -> TuningResult:
        space, max_evals, algo = build_hyperopt_space(model_config)

        def _safe_objective(args):
            try:
                return coordinator.objective(to_native(args))
            except Exception as exc:
                coordinator.logger.exception("Hyperopt trial failed", exc_info=exc)
                return {
                    "loss": np.inf,
                    "status": STATUS_FAIL,
                    "exception": repr(exc),
                }

        if algo == OptimizationAlgorithm.GRID.value:
            trials = self._grid_search(_safe_objective, space)
        else:
            trials = self._search(_safe_objective, space, max_evals, algo)

        # Models are kept only for the best trial, so pop them before copying it
        models = {trial["tid"]: trial["result"].pop("best_model", None) for trial in trials.trials}
        best_trial = self._select_best_trial(trials)
        if best_trial is None:
            return TuningResult(
                trials=trials,
                best_model=None,
                best_params={},
                best_loss=np.inf,
                best_trial=None,
                best_raw=None,
            )

        best_raw = {k: v[0] for k, v in best_trial["misc"]["vals"].items() if v}
        return TuningResult(
            trials=trials,
            best_model=models[best_trial["tid"]],
            best_params=to_native(space_eval(space, best_raw)),
            best_loss=best_trial["result"]["loss"],
            best_trial=best_trial,
            best_raw=best_raw,
        )

    def _search(self, objective, space, max_evals: int, algo) -> Trials:
        trials = Trials()
        try:
            fmin(
                fn=objective,
                space=space,
                algo=algo,
                trials=trials,
                rstate=self._rstate,
                max_evals=max_evals,
                show_progressbar=self._show_progressbar,
            )
        except AllTrialsFailed:
            # Raised by `fmin` only when returning the best point: the failed trials stay recorded
            pass
        return trials

    def _grid_search(self, objective, space) -> Trials:
        names = list(space.keys())
        choices = [self._extract_choices(name, node) for name, node in space.items()]
        # Each grid point is a tuple of choice indices, recorded in the trial as hyperopt does
        grid = list(product(*(range(len(values)) for values in choices)))

        trials = Trials()
        tids = trials.new_trial_ids(len(grid))
        results = []
        miscs = []
        for tid, indices in zip(tids, grid):
            params = {name: values[i] for name, values, i in zip(names, choices, indices)}
            results.append(objective(params))
            miscs.append({
                "tid": tid,
                "cmd": None,
                "idxs": {name: [tid] for name in names},
                "vals": {name: [i] for name, i in zip(names, indices)},
            })

        docs = trials.new_trial_docs(tids, [None] * len(grid), results, miscs)
        for doc in docs:
            doc["state"] = JOB_STATE_DONE
        trials.insert_trial_docs(docs)
        trials.refresh()
        return trials

    @staticmethod
    def _select_best_trial(trials: Trials) -> Optional[dict]:
        valid_trials = []
        for trial in trials.trials:
            result = trial.get("result", {})
            if result.get("status") != STATUS_OK:
                continue
            loss = result.get("loss", np.inf)
            if not np.isfinite(loss):
                continue
            valid_trials.append(trial)
        if not valid_trials:
            return None
        return copy.deepcopy(min(valid_trials, key=lambda t: t["result"]["loss"]))

    @staticmethod
    def _extract_choices(name: str, node) -> list:
        if getattr(node, "name", None) != "switch":
            raise ValueError(
                "Grid search supports only discrete choices (hp.choice or explicit lists). "
                f"Parameter '{name}' is not a choice."
            )
        # Choices are constant expressions (literals, tuples, ...): evaluate them as they are
        return [rec_eval(arg) for arg in node.pos_args[1:]]


OPT_ALGORITHMS = {
    OptimizationAlgorithm.TPE: tpe.suggest,
    OptimizationAlgorithm.ATPE: atpe.suggest,
    # Mixture suggested by hyperopt's documentation (`mix.suggest` has no default for it)
    OptimizationAlgorithm.MIX: partial(
        mix.suggest, p_suggest=[(0.1, rand.suggest), (0.2, anneal.suggest), (0.7, tpe.suggest)]
    ),
    OptimizationAlgorithm.RAND: rand.suggest,
    OptimizationAlgorithm.ANNEAL: anneal.suggest,
    OptimizationAlgorithm.GRID: OptimizationAlgorithm.GRID.value
}


def build_hyperopt_space(config: RecommenderConfig):
    space_list = []
    uses_distributions = False
    opt_alg_name = config.meta.hyper_opt_alg
    model_params = config.model_dump()
    # Unknown (extra) fields are not hyperparameters of the model: they are not searched
    model_fields = type(config).model_fields

    for k, value in model_params.items():
        if k not in model_fields or not isinstance(value, list):
            continue
        if value[0] not in SearchSpace:
            raise ValueError(
                f"Parameter '{k}' is not a search space: its first element must be one of "
                f"{[s.value for s in SearchSpace]}."
            )
        func_ = getattr(hp, value[0])
        val_items = value[1:]
        if value[0] == SearchSpace.CHOICE.value:
            space_list.append((k, func_(k, val_items)))
        else:
            uses_distributions = True
            space_list.append((k, func_(k, *val_items)))

    space = OrderedDict(space_list)

    # Number of combinations of the discrete choices (distributions have no finite size)
    grid_size = reduce(
        lambda x, y: x * y,
        [len(param.pos_args) - 1 for param in space.values() if param.name == "switch"],
        1
    )
    hyper_max_evals = config.meta.hyper_max_evals

    if uses_distributions:
        if opt_alg_name == OptimizationAlgorithm.GRID:
            raise ValueError(
                "Grid search supports only discrete choices. "
                "Please use explicit lists or 'choice' in model.meta configuration."
            )
        if hyper_max_evals is None:
            raise ValueError(
                "`hyper_max_evals` must be provided when using hyperopt distributions. "
                "Please define `hyper_max_evals` in model.meta configuration."
            )

    max_evals = hyper_max_evals if hyper_max_evals is not None else grid_size

    if max_evals <= 0:
        raise ValueError("`hyper_max_evals` must be a positive integer in model.meta configuration.")

    if opt_alg_name == OptimizationAlgorithm.GRID and max_evals < grid_size:
        raise ValueError(
            "`hyper_max_evals` must be >= the full grid size when using grid search."
        )

    opt_alg = OPT_ALGORITHMS[config.meta.hyper_opt_alg]

    return space, max_evals, opt_alg
