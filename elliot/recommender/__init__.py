"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from .base_recommender_model import BaseRecommenderModel

from .latent_factor_models import BPRMF, BPRMF_batch, WRMF, PureSVD, MF, FunkSVD, PMF, LMF, NonNegMF, FM, LogisticMF, \
    BPRSlim, Slim, CML, FISM
from .unpersonalized import Random, MostPop
from .autoencoders import MultiDAE, MultiVAE
from .knowledge_aware import KaHFM, KaHFMBatch, KaHFMEmbeddings
from .graph_based import NGCF, LightGCN
from .visual_recommenders import VBPR, DeepStyle, ACF
from .NN import ItemKNN, UserKNN, AttributeItemKNN, AttributeUserKNN
from .neural import DMF, NeuMF, GMF, NAIS, UserAutoRec, ItemAutoRec, ConvNeuMF
from .content_based import VSM
from .algebric import SlopeOne
from .adversarial import AMF, AMR
from .gan import IRGAN, CFGAN
# from .attentive import AFM
