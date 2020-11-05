"""
"""

__version__ = '0.1'
__author__ = 'XXX'


from .model_coordinator import ModelCoordinator
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

