"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import scipy.sparse as sp
from tqdm import tqdm

from elliot.utils.write import store_recommendation

import numpy as np
import random

from elliot.dataset.samplers import custom_sampler as cs

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.recommender_utils_mixin import RecMixin

from elliot.recommender.graph_based.lightgcn.LightGCN_model import LightGCNModel
from elliot.recommender.base_recommender_model import init_charger


class LightGCN(RecMixin, BaseRecommenderModel):
    r"""
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3397271.3401063>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        n_layers: Number of embedding propagation layers
        n_fold: Number of folds to split the adjacency matrix into sub-matrices and ease the computation

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        LightGCN:
          meta:
            save_recs: True
          lr: 0.0005
          epochs: 50
          batch_size: 512
          factors: 64
          batch_size: 256
          l_w: 0.1
          n_layers: 1
          n_fold: 5
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        """

        self._ratings = self._data.train_dict
        self._sampler = cs.Sampler(self._data.i_train_dict)

        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, None, None),
            ("_factors", "latent_dim", "factors", 64, None, None),
            ("_n_layers", "n_layers", "n_layers", 1, None, None),
            ("_l_w", "l_w", "l_w", 0.1, None, None),
            ("_n_fold", "n_fold", "n_fold", 1, None, None)
        ]
        self.autoset_params()

        self._adjacency, self._laplacian = self._create_adj_mat()

        self._model = LightGCNModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            n_layers=self._n_layers,
            l_w=self._l_w,
            n_fold=self._n_fold,
            adjacency=self._adjacency,
            laplacian=self._laplacian,
            random_seed=self._seed
        )

    def _create_adj_mat(self):
        adjacency = sp.dok_matrix((self._num_users + self._num_items,
                                   self._num_users + self._num_items), dtype=np.float32)
        adjacency = adjacency.tolil()
        ratings = self._data.sp_i_train.tolil()

        adjacency[:self._num_users, self._num_users:] = ratings
        adjacency[self._num_users:, :self._num_users] = ratings.T
        adjacency = adjacency.todok()

        def normalized_adj_bi(adj):
            # This is exactly how it's done in the paper. Different normalization approaches might be followed.
            rowsum = np.array(adj.sum(1))
            rowsum += 1e-7  # to avoid division by zero warnings

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            bi_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return bi_adj.tocoo()

        laplacian = normalized_adj_bi(adjacency)

        return adjacency.tocsr(), laplacian.tocsr()

    @property
    def name(self):
        return "LightGCN" \
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
