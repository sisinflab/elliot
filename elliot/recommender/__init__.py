"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from .base_recommender_model import BaseRecommenderModel

from .latent_factor_models import BPRMF, NNBPRMF, WRMF, PureSVD
from .unpersonalized import Random, MostPop
from .visual_recommenders import VBPR
# from .adversarial import APR, AMR
from .autoencoders import MultiDAE
from .autoencoders import MultiVAE
from .knowledge_aware import KaHFM
from .knowledge_aware import KaHFMBatch
from .graph_based import NGCF
from .NN import ItemKNN, UserKNN
from .neural import NeuralMatrixFactorization as NeuMF
