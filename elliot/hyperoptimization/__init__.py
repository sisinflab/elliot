"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from elliot.hyperoptimization.engine import HyperOptEngine, TuningResult
from elliot.hyperoptimization.model_coordinator import ModelCoordinator
from elliot.hyperoptimization.policy import EvaluationPolicy, FinalPolicy, SearchPolicy
from elliot.hyperoptimization.runner import HyperoptRunner, RunOutcome
from hyperopt import anneal, atpe, mix, rand, tpe

GRID_ALGO = "grid"


def parse_algorithms(opt_alg):
    if opt_alg not in _optimization_algorithms:
        raise KeyError(
            f"Unknown hyperopt algorithm '{opt_alg}'. "
            f"Available algorithms: {', '.join(sorted(_optimization_algorithms.keys()))}"
        )
    return _optimization_algorithms[opt_alg]



_optimization_algorithms = {
    "tpe": tpe.suggest,
    "atpe": atpe.suggest,
    "mix": mix.suggest,
    "rand": rand.suggest,
    "anneal": anneal.suggest,
    "grid": GRID_ALGO,
}
