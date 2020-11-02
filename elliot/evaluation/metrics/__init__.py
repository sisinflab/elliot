"""
This is the metrics' module.

This module contains and expose the recommendation metrics.
Each metric is encapsulated in a specific package.

See the implementation of Precision metric for creating new per-user metrics.
See the implementation of Item Coverage for creating new cross-user metrics.
"""

__version__ = '0.1'
__author__ = 'XXX'


from .precision import Precision
from .recall import Recall
from .item_coverage import ItemCoverage
import re

_metric_dictionary = {
    "Precision": Precision,
    "Recall": Recall,
    "ItemCoverage": ItemCoverage
}


def parse_metrics(metrics):
    return [_metric_dictionary[m] for m in metrics if m in _metric_dictionary.keys()]
