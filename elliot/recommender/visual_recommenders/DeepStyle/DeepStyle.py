"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import logging as log
from utils import logging
import os

import numpy as np
import tensorflow as tf

from elliot.recommender.latent_factor_models.NNBPRMF.NNBPRMF import NNBPRMF
from elliot.recommender.visual_recommenders.DeepStyle.DeepStyle_model import DeepStyle_model
from elliot.recommender.base_recommender_model import init_charger
np.random.seed(0)
tf.random.set_seed(0)
log.disable(log.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DeepStyle(NNBPRMF):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        np.random.seed(42)

        self.autoset_params()

        item_indices = [self._data.item_mapping[self._data.private_items[item]] for item in range(self._num_items)]

        self._model = DeepStyle_model(self._params.embed_k,
                                      self._params.lr,
                                      self._params.l_w,
                                      self._data.visual_features[item_indices],
                                      self._data.visual_features.shape[1],
                                      self._num_users,
                                      self._num_items)

    @property
    def name(self):
        return "DeepStyle" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

