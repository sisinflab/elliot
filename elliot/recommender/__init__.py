"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from .base_recommender_model import BaseRecommenderModel
from .keras_base_recommender_model import RecommenderModel

from .latent_factor_models import BPRMF, NNBPRMF
from .unpersonalized import Random
from .visual_recommenders import VBPR
from .adversarial import APR, AMR
from .autoencoders import MultiDAE
