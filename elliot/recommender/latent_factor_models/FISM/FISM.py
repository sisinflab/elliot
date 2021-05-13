"""
Module description:

"""

__version__ = '0.2'
__author__ = 'Zhankui (Aaron) He, Vito Walter Anelli, Claudio Pomo, Felice Antonio Merra'
__email__ = 'zkhe15@fudan.edu.cn, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, felice.merra@poliba.it'
__paper__ = 'FISM: Factored Item Similarity Models for Top-N Recommender Systems by Santosh Kabbur, Xia Ning, and George Karypis'

import numpy as np
from tqdm import tqdm
import pickle

from elliot.dataset.samplers import pointwise_pos_neg_ratio_ratings_sampler as pws
from elliot.utils.write import store_recommendation

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.latent_factor_models.FISM.FISM_model import FISM_model
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.base_recommender_model import init_charger


class FISM(RecMixin, BaseRecommenderModel):
    r"""
    FISM: Factored Item Similarity Models

    For further details, please refer to the `paper <http://glaros.dtc.umn.edu/gkhome/node/1068>`_

    Args:
        factors: Number of factors of feature embeddings
        lr: Learning rate
        beta: Regularization coefficient for latent factors
        lambda: Regularization coefficient for user bias
        gamma: Regularization coefficient for item bias
        alpha: Alpha parameter (a value between 0 and 1)
        neg_ratio: ratio of sampled negative items

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        FISM:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 10
          lr: 0.001
          alpha: 0.5
          beta: 0.001
          lambda: 0.001
          gamma: 0.001
          neg_ratio: 0.5
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a FISM instance.
        (see http://glaros.dtc.umn.edu/gkhome/node/1068 for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {_factors: embedding size,
                                      [l_w, l_b]: regularization,
                                      lr: learning rate}
        """

        self._params_list = [
            ("_factors", "factors", "factors", 100, None, None),
            ("_lr", "lr", "lr", 0.001, None, None),
            ("_beta", "beta", "beta", 0.001, None, None),
            ("_lambda", "lambda", "lambda", 0.001, None, None),
            ("_gamma", "gamma", "gamma", 0.001, None, None),
            ("_alpha", "alpha", "alpha", 0.5, lambda x: min(max(0, x), 1), None),
            ("_neg_ratio", "neg_ratio", "neg_ratio", 0.5, None, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._sampler = pws.Sampler(self._data.i_train_dict, self._data.sp_i_train_ratings, self._neg_ratio)

        self._model = FISM_model(self._data,
                                 self._factors,
                                 self._lr,
                                 self._alpha,
                                 self._beta,
                                 self._lambda,
                                 self._gamma,
                                 self._num_users,
                                 self._num_items,
                                 self._seed)

    @property
    def name(self):
        return "FISM" \
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


