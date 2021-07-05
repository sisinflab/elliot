"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import numpy as np
import pickle
from tqdm import tqdm

from elliot.dataset.samplers import pointwise_pos_neg_sampler as pws
from elliot.recommender.latent_factor_models.LogisticMF.logistic_matrix_factorization_model import LogisticMatrixFactorizationModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger


class LogisticMatrixFactorization(RecMixin, BaseRecommenderModel):
    r"""
    Logistic Matrix Factorization

    For further details, please refer to the `paper <https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf>`_

    Args:
        factors: Number of factors of feature embeddings
        lr: Learning rate
        reg: Regularization coefficient
        alpha: Parameter for confidence estimation

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        LogisticMatrixFactorization:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 10
          lr: 0.001
          reg: 0.1
          alpha: 0.5
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_factors", "factors", "factors", 10, None, None),
            ("_l_w", "reg", "reg", 0.1, None, None),
            ("_alpha", "alpha", "alpha", 0.5, None, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._sampler = pws.Sampler(self._data.i_train_dict)

        self._model = LogisticMatrixFactorizationModel(self._num_users,
                                                       self._num_items,
                                                       self._factors,
                                                       self._l_w,
                                                       self._alpha,
                                                       self._learning_rate,
                                                       self._seed)

    @property
    def name(self):
        return "LMF"\
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def predict(self, u: int, i: int):
        pass

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions * 2 // self._batch_size), disable=not self._verbose) as t:
                # update items and fix users
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    self._model.set_update_user(False)
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

                # update users and fix items
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    self._model.set_update_user(True)
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss.numpy()/(it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict_batch(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test


