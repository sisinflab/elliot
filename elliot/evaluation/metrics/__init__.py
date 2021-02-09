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

from evaluation.metrics.accuracy.ndcg import NDCG
from evaluation.metrics.accuracy.precision import Precision
from evaluation.metrics.accuracy.recall import Recall
from evaluation.metrics.accuracy.hit_rate import HR
from evaluation.metrics.accuracy.mrr import MRR
from evaluation.metrics.accuracy.map import MAP
from evaluation.metrics.accuracy.mar import MAR
from evaluation.metrics.accuracy.f1 import F1
from evaluation.metrics.accuracy.DSC import DSC
from evaluation.metrics.accuracy.AUC import LAUC, AUC, GAUC

from evaluation.metrics.rating.mae import MAE
from evaluation.metrics.rating.mse import MSE
from evaluation.metrics.rating.rmse import RMSE

from evaluation.metrics.coverage import ItemCoverage, UserCoverage, NumRetrieved

from evaluation.metrics.diversity.gini_index import GiniIndex
from evaluation.metrics.diversity.shannon_entropy import ShannonEntropy
from evaluation.metrics.diversity.SRecall import SRecall

from evaluation.metrics.novelty.EFD import EFD
from evaluation.metrics.novelty.EPC import EPC

from evaluation.metrics.bias import ARP, APLT, ACLT, PopRSP, PopREO

from evaluation.metrics.fairness.MAD import UserMADrating, ItemMADrating, UserMADranking, ItemMADranking
from evaluation.metrics.fairness.BiasDisparity import BiasDisparityBR, BiasDisparityBS, BiasDisparityBD
from evaluation.metrics.fairness.rsp import RSP
from evaluation.metrics.fairness.reo import REO

from evaluation.metrics.statistical_array_metric import StatisticalMetric

_metric_dictionary = {
    "nDCG": NDCG,
    "Precision": Precision,
    "Recall": Recall,
    "HR": HR,
    "MRR": MRR,
    "MAP": MAP,
    "MAR": MAR,
    "F1": F1,
    "DSC": DSC,
    "LAUC": LAUC,
    "GAUC": GAUC,
    "AUC": AUC,
    "ItemCoverage": ItemCoverage,
    "UserCoverage": UserCoverage,
    "NumRetrieved": NumRetrieved,
    "Gini": GiniIndex,
    "SEntropy": ShannonEntropy,
    "EFD": EFD,
    "EPC": EPC,
    "MAE": MAE,
    "MSE": MSE,
    "RMSE": RMSE,
    "UserMADrating": UserMADrating,
    "ItemMADrating": ItemMADrating,
    "UserMADranking": UserMADranking,
    "ItemMADranking": ItemMADranking,
    "BiasDisparityBR": BiasDisparityBR,
    "BiasDisparityBS": BiasDisparityBS,
    "BiasDisparityBD": BiasDisparityBD,
    "SRecall": SRecall,
    "ARP": ARP,
    "APLT": APLT,
    "ACLT": ACLT,
    "PopRSP": PopRSP,
    "PopREO": PopREO,
    "RSP": RSP,
    "REO": REO
}

_lower_dict = {k.lower(): v for k, v in _metric_dictionary.items()}


def parse_metrics(metrics):
    return [_lower_dict[m.lower()] for m in metrics if m.lower() in _lower_dict.keys()]


def parse_metric(metric):
    metric = metric.lower()
    return _lower_dict[metric] if metric in _lower_dict.keys() else None
