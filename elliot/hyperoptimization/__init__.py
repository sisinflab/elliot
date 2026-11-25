"""
Module description:

"""


from typing import Any, Dict

import numpy as np
from hyperopt import anneal


def patch_hyperopt():
    """Patch hyperopt in place. Idempotent: patched functions are applied only once."""
    _patch_anneal()


def to_native(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return `params` with every numpy scalar value converted to its Python equivalent."""
    return {k: v.item() if isinstance(v, np.generic) else v for k, v in params.items()}


def _patch_anneal():
    # hyperopt 0.2.7's `AnnealingAlgo.choose_ltv` casts a 1-element array with `int()`, an error
    # with numpy >= 2 (fixed in 0.3.0): replicate its single-draw branch, delegating the draw
    # itself as a sized one. On 0.3.0 this behaves exactly as the original
    original = anneal.AnnealingAlgo.choose_ltv
    if getattr(original, "_elliot_patched", False):
        return

    def choose_ltv(self, label, size):
        if size != 1:
            return original(self, label, size)
        tids = self.node_tids[label]
        for tid in self.best_tids:
            if tid in tids:
                return self.tid_losses_dct[tid], tid, self.node_vals[label][tids.index(tid)]
        loss, tid, val = original(self, label, (1,))
        self.best_tids.append(int(tid[0]))
        return loss, tid, val

    choose_ltv._elliot_patched = True
    anneal.AnnealingAlgo.choose_ltv = choose_ltv


# Submodules import the helpers above from the package: keep these imports after them
from .engine import HyperOptEngine, TuningResult
from .model_coordinator import ModelCoordinator
from .runner import (
    RunOutcome,
    run_hyperopt,
    run_single,
    run_proxy,
    run_evaluation,
    requires_hyperopt
)
