"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import scipy.sparse as sp
from tqdm import tqdm

from ast import literal_eval as make_tuple
from utils.write import store_recommendation

import numpy as np
import random
from utils import logging

from dataset.samplers import custom_sampler as cs
from evaluation.evaluator import Evaluator
from utils.folder import build_model_folder

from recommender import BaseRecommenderModel
from recommender.recommender_utils_mixin import RecMixin

from recommender.graph_based.ngcf.NGCFModel import NGCFModel

np.random.seed(42)
random.seed(0)


class NGCF(RecMixin, BaseRecommenderModel):

    def __init__(self, data, config, params, *args, **kwargs):
        """
        """
        super().__init__(data, config, params, *args, **kwargs)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random
        self._random_p = random
        self._num_iters = self._params.epochs

        self._ratings = self._data.train_dict
        self._sampler = cs.Sampler(self._data.i_train_dict)
        self._iteration = 0
        self.evaluator = Evaluator(self._data, self._params)
        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params.name = self.name

        self._learning_rate = self._params.learning_rate
        self._embed_k = self._params.embed_k
        self._l_w = self._params.l_w
        self._weight_size = list(make_tuple(self._params.weight_size))
        self._n_layers = len(self._weight_size)
        self._node_dropout = list(make_tuple(self._params.node_dropout))
        self._message_dropout = list(make_tuple(self._params.message_dropout))
        self._n_fold = self._params.n_fold

        self._plain_adj, self._norm_adj, self._mean_adj = self.create_adj_mat()

        self._model = NGCFModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._embed_k,
            l_w=self._l_w,
            weight_size=self._weight_size,
            n_layers=self._n_layers,
            node_dropout=self._node_dropout,
            message_dropout=self._message_dropout,
            n_fold=self._n_fold,
            plain_adj=self._plain_adj,
            norm_adj=self._norm_adj,
            mean_adj=self._mean_adj
        )

        build_model_folder(self._config.path_output_rec_weight, self.name)
        self._saving_filepath = f'{self._config.path_output_rec_weight}{self.name}/best-weights-{self.name}'
        self.logger = logging.get_logger(self.__class__.__name__)

    def create_adj_mat(self):
        adj_mat = sp.dok_matrix((self._num_users + self._num_items,
                                 self._num_users + self._num_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self._data.sp_i_train.tolil()

        adj_mat[:self._num_users, self._num_users:] = R
        adj_mat[self._num_users:, :self._num_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_bi(adj):
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            bi_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return bi_adj.tocoo()

        norm_adj_mat = normalized_adj_bi(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_bi(adj_mat)

        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    @property
    def name(self):
        return "NGCF" \
               + "_lr:" + str(self._params.learning_rate) \
               + "-embedk:" + str(self._params.embed_k) \
               + "-lw:" + str(self._params.l_w) \
               + "-weightsize:" + str(self._params.weight_size) \
               + "-nlayers:" + str(len(self._params.weight_size)) \
               + "-nodedropout:" + str(self._params.node_dropout) \
               + "-messagedropout:" + str(self._params.message_dropout) \
               + "-nfold:" + str(self._params.n_fold)

    def train(self):
        best_metric_value = 0
        for it in range(self._num_iters):
            self.restore_weights(it)
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in zip(*self._sampler.step(self._data.transactions, self._batch_size)):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            if not (it + 1) % self._validation_rate:
                recs = self.get_recommendations(self._config.top_k)
                results, statistical_results = self.evaluator.eval(recs)
                self._results.append(results)
                self._statistical_results.append(statistical_results)
                print(f'Epoch {(it + 1)}/{self._num_iters} loss {loss:.3f}')

                if self._results[-1][self._validation_metric] > best_metric_value:
                    print("******************************************")
                    best_metric_value = self._results[-1][self._validation_metric]
                    if self._save_weights:
                        self._model.save_weights(self._saving_filepath)
                    if self._save_recs:
                        store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        for index, offset in enumerate(range(0, self._num_users, self._params.batch_size)):
            offset_stop = min(offset+self._params.batch_size, self._num_users)
            predictions = self._model.predict(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, self.get_train_mask(offset, offset_stop), k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
        return predictions_top_k

