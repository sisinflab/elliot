"""
This is the metrics' module.

This module contains and expose the recommendation metrics.
Each metric is encapsulated in a specific package.

See the implementation of Precision metric for creating new per-user metrics.
See the implementation of Item Coverage for creating new cross-user metrics.
"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from .ndcg import NDCG
from .precision import Precision
from .recall import Recall
from .item_coverage import ItemCoverage
from .statistical_array_metric import StatisticalMetric
from .MAD import UserMADrating
from .hit_rate import HR
from .mrr import MRR
from .map import MAP
from .mae import MAE
from .mse import MSE
from .rmse import RMSE
from .f1 import F1
from .DSC import DSC
from .gini_index import GiniIndex
from .shannon_entropy import ShannonEntropy
from .EFD import EFD

_metric_dictionary = {
    "nDCG": NDCG,
    "Precision": Precision,
    "Recall": Recall,
    "ItemCoverage": ItemCoverage,
    "UserMADrating": UserMADrating,
    "HR": HR,
    "MRR": MRR,
    "MAP": MAP,
    "MAE": MAE,
    "MSE": MSE,
    "RMSE": RMSE,
    "F1": F1,
    "DSC": DSC,
    "Gini": GiniIndex,
    "SEntropy": ShannonEntropy,
    "EFD": EFD
}


def parse_metrics(metrics):
    return [_metric_dictionary[m] for m in metrics if m in _metric_dictionary.keys()]


def parse_metric(metric):
    return _metric_dictionary[metric] if metric in _metric_dictionary.keys() else None
