"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from .model_coordinator import ModelCoordinator
from .model_coordinator_bis import ModelCoordinator as ModelCoordinator2
from hyperopt import tpe, atpe, mix, rand, anneal

_optimization_algorithms = {
    "tpe": tpe.suggest,
    "atpe": atpe.suggest,
    "mix": mix.suggest,
    "rand": rand.suggest,
    "anneal": anneal.suggest
}


def parse_algorithms(opt_alg):
    return _optimization_algorithms[opt_alg]

