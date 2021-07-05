"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import numpy as np
from tqdm import tqdm

from elliot.dataset.samplers import pointwise_pos_neg_sampler as pws
from elliot.recommender.neural.GeneralizedMF.generalized_matrix_factorization_model import GeneralizedMatrixFactorizationModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger


class GMF(RecMixin, BaseRecommenderModel):
    r"""
        Neural Collaborative Filtering

        For further details, please refer to the `paper <https://arxiv.org/abs/1708.05031>`_

        Args:
            mf_factors: Number of latent factors
            lr: Learning rate
            is_edge_weight_train: Whether the training uses edge weighting


        To include the recommendation model, add it to the config file adopting the following pattern:

        .. code:: yaml

          models:
            GMF:
              meta:
                save_recs: True
              epochs: 10
              batch_size: 512
              mf_factors: 10
              lr: 0.001
              is_edge_weight_train: True
        """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._sampler = pws.Sampler(self._data.i_train_dict)

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_mf_factors", "mf_factors", "mffactors", 10, None, None),
            ("_is_edge_weight_train", "is_edge_weight_train", "isedgeweighttrain", True, None, None)
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))
        self._model = GeneralizedMatrixFactorizationModel(self._num_users, self._num_items,
                                                          self._mf_factors,
                                                          self._is_edge_weight_train,
                                                          self._learning_rate,
                                                          self._seed)

    @property
    def name(self):
        return "GeneralizedMF"\
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
            predictions = self._model.get_recs(
                (
                    np.repeat(np.array(list(range(offset, offset_stop)))[:, None], repeats=self._num_items, axis=1),
                    np.array([self._i_items_set for _ in range(offset, offset_stop)])
                )
            )
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)

            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

