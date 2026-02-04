"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from elliot.hyperoptimization.engine import HyperOptEngine, TuningResult
from elliot.hyperoptimization.model_coordinator import ModelCoordinator
from elliot.hyperoptimization.policy import EvaluationPolicy, FinalPolicy, SearchPolicy
from elliot.hyperoptimization.runner import RunOutcome, run_hyperopt
