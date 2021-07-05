"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pickle

from elliot.recommender.latent_factor_models.NonNegMF.non_negative_matrix_factorization_model import NonNegMFModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger


class NonNegMF(RecMixin, BaseRecommenderModel):
    r"""
    Non-Negative Matrix Factorization

    For further details, please refer to the `paper <https://ieeexplore.ieee.org/document/6748996>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        reg: Regularization coefficient

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        NonNegMF:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 10
          lr: 0.001
          reg: 0.1
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_factors", "factors", "factors", 10, None, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_l_w", "reg", "reg", 0.1, None, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict
        self._global_mean = np.mean(self._data.sp_i_train_ratings)
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._model = NonNegMFModel(self._data,
                                    self._num_users,
                                    self._num_items,
                                    self._global_mean,
                                    self._factors,
                                    self._l_w,
                                    self._learning_rate,
                                    random_seed=self._seed)

    @property
    def name(self):
        return "NonNegMF" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_recs(u, mask, k) for u in self._data.train_dict.keys()}


    def train(self):
        print(f"Transactions: {self._data.transactions}")
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            print(f"\n********** Iteration: {it + 1}")
            self._iteration = it

            self._model.train_step()

            self.evaluate(it)

