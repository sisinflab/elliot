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


def parse_metrics(strings):
    """
    This function parse the string provided by the user and creates a list of classes for the metrics' computation.

    Available strings:
    *
    Precision,
    Recall,
    ItemCoverage
    *

    :param strings: a string containing the names of the metrics in the form '[Precision,...]'
    :return: a list of metric classes
    """
    if (strings[0] != "[") | (strings[-1] != "]"):
        raise SyntaxError("Not a valid list")
    temp = strings[1:-1]
    temp = re.sub(r"\s+", "", temp)
    temp = temp.split(",")
    return [_metric_dictionary[m] for m in temp if m in _metric_dictionary.keys()]
