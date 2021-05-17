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

from elliot.lp_evaluation.metrics.accuracy.ndcg import nDCG
from elliot.lp_evaluation.metrics.accuracy.precision import Precision
from elliot.lp_evaluation.metrics.accuracy.recall import Recall
from elliot.lp_evaluation.metrics.accuracy.hit_rate import HR
from elliot.lp_evaluation.metrics.accuracy.mrr import MRR
from elliot.lp_evaluation.metrics.accuracy.mrr import MRRAtN
from elliot.lp_evaluation.metrics.accuracy.ap import AP
from elliot.lp_evaluation.metrics.accuracy.auc import AUC

_metric_dictionary = {
    "nDCG": nDCG,
    "Precision": Precision,
    "Recall": Recall,
    "HR": HR,
    "MRR": MRR,
    "MRRAtN": MRRAtN,
    "AP": AP,
    "AUC": AUC,
}

_lower_dict = {k.lower(): v for k, v in _metric_dictionary.items()}


def parse_metrics(metrics):
    return [_lower_dict[m.lower()] for m in metrics if m.lower() in _lower_dict.keys()]


def parse_metric(metric):
    metric = metric.lower()
    return _lower_dict[metric] if metric in _lower_dict.keys() else None
