"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from .BPRMF_batch import BPRMF_batch
from .BPRMF import BPRMF
from .wrmf import WRMF
from .pure_svd import PureSVD
from .mf import MF
from .funk_svd import FunkSVD
from .pmf import PMF
from .logistic_mf import LogisticMF
from .non_neg_mf import NonNegMF
from .FM import FM
from .FMnofeatures import FMnofeatures
from .FFM import FFM
from .BPRSlim import BPRSlim
from .Slim import Slim
from .CML import CML
from .FISM import FISM
from .svdpp import SVDpp
from .ials import iALS
from .mf2020 import MF2020, MF2020Batch