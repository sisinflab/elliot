"""
This is the metrics' module.

This module contains and expose the recommendation metrics.
"""

__version__ = '0.1'
__author__ = 'XXX'


from .latent_factor_models import BPRMF, NNBPRMF
from .unpersonalized import Random
from .visual_recommenders import VBPR
from .adversarial import APR, AMR
