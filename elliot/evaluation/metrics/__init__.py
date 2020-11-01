"""
This is the metrics' module.

This module contains and expose the recommendation metrics.
"""

__version__ = '0.1'
__author__ = 'XXX'


from .Precision import Precision
from .Recall import Recall
import re

_metric_dictionary = {
    "Precision": Precision,
    "Recall": Recall
}

def parse_metrics(strings):
    if (strings[0] != "[") | (strings[-1] != "]"):
        raise SyntaxError("Not a valid list")
    temp = strings[1:-1]
    temp = re.sub(r"\s+", "", temp)
    temp = temp.split(",")
    return [_metric_dictionary[m] for m in temp if m in _metric_dictionary.keys()]
