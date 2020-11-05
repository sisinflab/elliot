"""
This is the evaluation module.

This module contains and expose the recommendation metrics.
It contains the evaluator object that should be initialized in the __init__ of the recommendation system,
and called at the end of each training step.
"""

__version__ = '0.1'
__author__ = 'XXX'

from . import metrics
