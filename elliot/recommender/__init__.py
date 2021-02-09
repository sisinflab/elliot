"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from .base_recommender_model import BaseRecommenderModel

from .latent_factor_models import BPRMF, NNBPRMF, WRMF, PureSVD, MF, FunkSVD, PMF, LMF, NonNegMF, FM, LogisticMF, BPRSlim
from .unpersonalized import Random, MostPop
# from .adversarial import APR, AMR
from .autoencoders import MultiDAE
from .autoencoders import MultiVAE
from .knowledge_aware import KaHFM, KaHFMBatch, KaHFMEmbeddings
from .graph_based import NGCF, LightGCN
from .visual_recommenders import VBPR
from .NN import ItemKNN, UserKNN, AttributeItemKNN, AttributeUserKNN
from .neural import DeepMatrixFactorization as DMF, NeuralMatrixFactorization as NeuMF
from .content_based import VSM
from .algebric import SlopeOne
# from .attentive import AFM
