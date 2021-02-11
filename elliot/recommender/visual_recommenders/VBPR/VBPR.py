"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import logging
import os

import numpy as np
import tensorflow as tf

from recommender.visual_recommenders.visual_mixins.visual_loader_mixin import VisualLoader
from recommender.latent_factor_models.NNBPRMF.NNBPRMF import NNBPRMF
from recommender.visual_recommenders.VBPR.VBPR_model import VBPR_model

np.random.seed(0)
tf.random.set_seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class VBPR(NNBPRMF, VisualLoader):

    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a VBPR instance.
        (see https://arxiv.org/pdf/1510.01784.pdf for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w, l_b]: regularization,
                                      lr: learning rate}
        """
        super().__init__(data, config, params, *args, **kwargs)
        np.random.seed(42)

        self._l_e = self._params.l_e
        self._embed_d = self._params.embed_d

        self._params_list += [
            ("_embed_d", "embed_d", "embed_d", 20, None, None),
            ("_l_e", "l_e", "l_e", 0.1, None, None)
        ]

        self._params.name = self.name

        self.autoset_params()

        item_indices = [self._data.item_mapping[self._data.private_items[item]] for item in range(self._num_items)]

        self._model = VBPR_model(self._params.embed_k,
                                 self._params.embed_d,
                                 self._params.lr,
                                 self._params.l_w,
                                 self._params.l_b,
                                 self._params.l_e,
                                 self._data.visual_features[item_indices],
                                 self._data.visual_features.shape[1],
                                 self._num_users,
                                 self._num_items)

    @property
    def name(self):
        return "VBPR" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

