"""
Hyperopt optimization engine for Elliot.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Optional, Tuple

import numpy as np
from hyperopt import Trials, STATUS_FAIL, STATUS_OK, fmin, space_eval
from hyperopt.base import JOB_STATE_DONE


@dataclass(frozen=True)
class TuningResult:
    trials: Trials
    best_params: Dict[str, Any]
    best_loss: float
    best_trial: Optional[dict]
    best_raw: Optional[dict]


class HyperOptEngine:
    def __init__(self, rstate: Optional[np.random.Generator] = None, show_progressbar: bool = False):
        self._rstate = rstate
        self._show_progressbar = show_progressbar

    def optimize(self, coordinator, space, algo, max_evals: int) -> TuningResult:
        if algo == "grid":
            return self._grid_search(coordinator, space, max_evals)

        trials = Trials()

        def _safe_objective(args):
            try:
                return coordinator.objective(args)
            except Exception as exc:
                coordinator.logger.exception("Hyperopt trial failed", exc_info=exc)
                return {
                    "loss": np.inf,
                    "status": STATUS_FAIL,
                    "exception": repr(exc),
                }

        best_raw = fmin(
            fn=_safe_objective,
            space=space,
            algo=algo,
            trials=trials,
            rstate=self._rstate,
            max_evals=max_evals,
            show_progressbar=self._show_progressbar,
        )

        best_trial = self._select_best_trial(trials)
        best_params = space_eval(space, best_raw) if best_raw is not None else {}
        best_loss = best_trial["result"]["loss"] if best_trial is not None else np.inf

        return TuningResult(
            trials=trials,
            best_params=best_params,
            best_loss=best_loss,
            best_trial=best_trial,
            best_raw=best_raw,
        )

    def _grid_search(self, coordinator, space, max_evals: int) -> TuningResult:
        grid, choice_values = self._extract_grid(space)
        total = len(grid)
        if max_evals is not None and max_evals < total:
            raise ValueError(
                f"Grid search requires hyper_max_evals >= grid size "
                f"(hyper_max_evals={max_evals}, grid size={total})."
            )

        trials = Trials()
        tids = trials.new_trial_ids(total)
        results = []
        miscs = []
        params_by_tid = {}
        for tid, params in zip(tids, grid):
            params_by_tid[tid] = params
            result = coordinator.objective(params)
            results.append(result)
            miscs.append(self._build_misc(tid, params, choice_values))

        specs = [None] * total
        docs = trials.new_trial_docs(tids, specs, results, miscs)
        for doc in docs:
            doc["state"] = JOB_STATE_DONE
        trials.insert_trial_docs(docs)
        trials.refresh()

        best_trial = self._select_best_trial(trials)
        best_params = params_by_tid.get(best_trial["tid"]) if best_trial is not None else {}
        best_loss = best_trial["result"]["loss"] if best_trial is not None else np.inf

        return TuningResult(
            trials=trials,
            best_params=best_params,
            best_loss=best_loss,
            best_trial=best_trial,
            best_raw=best_params or None,
        )

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
        return min(valid_trials, key=lambda t: t["result"]["loss"])

    @staticmethod
    def _extract_grid(space) -> Tuple[list, Dict[str, list]]:
        if not space:
            return [{}], {}

        choice_values = {}
        for name, node in space.items():
            if getattr(node, "name", None) != "switch":
                raise ValueError(
                    "Grid search supports only discrete choices (hp.choice or explicit lists). "
                    f"Parameter '{name}' is not a choice."
                )
            values = [getattr(arg, "obj", None) for arg in node.pos_args[1:]]
            if any(v is None for v in values):
                raise ValueError(
                    f"Grid search supports only literal choices. "
                    f"Parameter '{name}' contains non-literal elements."
                )
            choice_values[name] = values

        keys = list(space.keys())
        grid = [dict(zip(keys, values)) for values in product(*(choice_values[k] for k in keys))]
        return grid, choice_values

    @staticmethod
    def _build_misc(tid: int, params: Dict[str, Any], choice_values: Dict[str, list]) -> Dict[str, Any]:
        vals = {}
        idxs = {}
        for key, value in params.items():
            if key in choice_values:
                try:
                    vals[key] = [choice_values[key].index(value)]
                except ValueError:
                    vals[key] = [value]
            else:
                vals[key] = [value]
            idxs[key] = [tid]
        return {"tid": tid, "cmd": None, "idxs": idxs, "vals": vals}
