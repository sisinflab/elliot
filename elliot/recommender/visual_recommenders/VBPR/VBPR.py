"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import logging
import os

import numpy as np
import tensorflow as tf

from dataset.datatype_mixins.visual_loader_mixin import VisualLoader
from recommender.attack.attack_visual_feature_mixin import AttackVisualFeature
from recommender.latent_factor_models.NNBPRMF.NNBPRMF import NNBPRMF
from recommender.visual_recommenders.VBPR.VBPR_model import VBPR_model

np.random.seed(0)
tf.random.set_seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class VBPR(NNBPRMF, VisualLoader, AttackVisualFeature):

    def __init__(self, config, params, *args, **kwargs):
        """
        Create a VBPR instance.
        (see https://arxiv.org/pdf/1510.01784.pdf for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w, l_b]: regularization,
                                      lr: learning rate}
        """
        super().__init__(config, params, *args, **kwargs)
        np.random.seed(42)

        self._embed_d = self._params.embed_d
        self._l_e = self._params.l_e

        self.process_visual_features(self._data)

        self._model = VBPR_model(self._params.embed_k,
                                 self._params.embed_d,
                                 self._params.lr,
                                 self._params.l_w,
                                 self._params.l_b,
                                 self._params.l_e,
                                 self._emb_image,
                                 self._num_image_feature,
                                 self._num_users,
                                 self._num_image_feature)

