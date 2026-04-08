"""
This is the metrics' module.

This module contains and expose the recommendation metrics.
Each metric is encapsulated in a specific package.

See the implementation of Precision metric for creating new per-user metrics.
See the implementation of Item Coverage for creating new cross-user metrics.
"""


from .accuracy.ndcg import nDCG, nDCGRendle2020
from .accuracy.precision import Precision
from .accuracy.recall import Recall
from .accuracy.hit_rate import HR
from .accuracy.mrr import MRR
from .accuracy.map import MAP
from .accuracy.mar import MAR
from .accuracy.f1 import F1, ExtendedF1
from .accuracy.DSC import DSC
from .accuracy.AUC import LAUC, AUC, GAUC

from .rating.mae import MAE
from .rating.mse import MSE
from .rating.rmse import RMSE

from .coverage import ItemCoverage, UserCoverage, NumRetrieved, UserCoverageAtN

from .diversity.gini_index import GiniIndex
from .diversity.shannon_entropy import ShannonEntropy
from .diversity.SRecall import SRecall

from .novelty.EFD import EFD, ExtendedEFD
from .novelty.EPC import EPC, ExtendedEPC

from .bias import ARP, APLT, ACLT, PopRSP, PopREO, ExtendedPopRSP, ExtendedPopREO

from .fairness.MAD import UserMADrating, ItemMADrating, UserMADranking, ItemMADranking
from .fairness.BiasDisparity import BiasDisparityBR, BiasDisparityBS, BiasDisparityBD
from .fairness.rsp import RSP
from .fairness.reo import REO

from .statistical_array_metric import StatisticalMetric

from .base_metric import BaseMetric
