"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.visual_recommenders.VBPR.VBPR_model import VBPR_model
from elliot.utils.write import store_recommendation


class VBPR(RecMixin, BaseRecommenderModel):
    r"""
    VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback

    For further details, please refer to the `paper <http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11914>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        factors_d: Dimension of visual factors
        batch_size: Batch size
        l_w: Regularization coefficient
        l_b: Regularization coefficient of bias
        l_e: Regularization coefficient of projection matrix

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        VBPR:
          meta:
            save_recs: True
          lr: 0.0005
          epochs: 50
          factors: 100
          factors_d: 20
          batch_size: 128
          l_w: 0.000025
          l_b: 0
          l_e: 0.002
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_factors", "factors", "factors", 100, None, None),
            ("_factors_d", "factors_d", "factors_d", 20, None, None),
            ("_learning_rate", "lr", "lr", 0.0005, None, None),
            ("_l_w", "l_w", "l_w", 0.000025, None, None),
            ("_l_b", "l_b", "l_b", 0, None, None),
            ("_l_e", "l_e", "l_e", 0.002, None, None),
            ("_loader", "loader", "load", "ItemAttributes", None, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._side = getattr(self._data.side_information, self._loader, None)

        self._sampler = cs.Sampler(self._data.i_train_dict)

        item_indices = [self._side.item_mapping[self._data.private_items[item]] for item in range(self._num_items)]

        self._model = VBPR_model(self._factors,
                                 self._factors_d,
                                 self._learning_rate,
                                 self._l_w,
                                 self._l_b,
                                 self._l_e,
                                 self._side.visual_features[item_indices],
                                 self._num_users,
                                 self._num_items,
                                 self._seed)

    @property
    def name(self):
        return "VBPR" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in range(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss.numpy())

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test


