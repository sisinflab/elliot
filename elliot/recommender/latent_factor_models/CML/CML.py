"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle

import numpy as np
from tqdm import tqdm

from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.latent_factor_models.CML.CML_model import CML_model
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation
from elliot.recommender.base_recommender_model import init_charger


class CML(RecMixin, BaseRecommenderModel):
    r"""
    Collaborative Metric Learning

    For further details, please refer to the `paper <https://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        l_w: Regularization coefficient for latent factors
        l_b: Regularization coefficient for bias
        margin: Safety margin size

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        CML:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 10
          lr: 0.001
          l_w: 0.001
          l_b: 0.001
          margin: 0.5
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a CML instance.
        (see https://vision.cornell.edu/se3/wp-content/uploads/2017/03/WWW-fp0554-hsiehA.pdf for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w, l_b]: regularization,
                                      lr: learning rate}
        """

        self._params_list = [
            ("_user_factors", "factors", "factors", 100, None, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_l_w", "l_w", "l_w", 0.001, None, None),
            ("_l_b", "l_b", "l_b", 0.001, None, None),
            ("_margin", "margin", "margin", 0.5, None, None),
        ]

        self.autoset_params()

        self._item_factors = self._user_factors

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._sampler = cs.Sampler(self._data.i_train_dict)

        self._model = CML_model(self._user_factors,
                                self._item_factors,
                                self._learning_rate,
                                self._l_w,
                                self._l_b,
                                self._margin,
                                self._num_users,
                                self._num_items,
                                self._seed)

    @property
    def name(self):
        return "CML" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss.numpy()/(it + 1))

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

