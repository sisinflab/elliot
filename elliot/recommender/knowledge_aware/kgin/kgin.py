"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Antonio Ferrara'
__email__ = 'antonio.ferrara@poliba.it'

import numpy as np
from tqdm import tqdm
import networkx as nx
from collections import defaultdict

from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.knowledge_aware.kaHFM_batch.tfidf_utils import TFIDF
from elliot.recommender.knowledge_aware.kahfm_embeddings.kahfm_embeddings_model import KaHFMEmbeddingsModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation


class KGIN(RecMixin, BaseRecommenderModel):
    """
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0001, None, None),
            ("_l_w", "l_w", "l_w", 0.005, None, None),
            ("_l_b", "l_b", "l_b", 0, None, None),
            ("_loader", "loader", "load", "ChainedKG", None, None),
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict

        self._side = getattr(self._data.side_information, self._loader, None)

        self._sampler = cs.Sampler(self._data.i_train_dict)

        ckg_graph = nx.MultiDiGraph()
        rd = dict()
        rd[0] = list(zip(*self._data.sp_i_train.nonzero()))

        print("\nBegin to load knowledge graph triples ...")
        for h_id, r_id, t_id in tqdm(self._side.triplets, ascii=True):
            ckg_graph.add_edge(h_id, t_id, key=r_id)
            rd[r_id].append([h_id, t_id])

        self._model = KaHFMEmbeddingsModel(self._user_factors,
                                           self._item_factors,
                                           self._params.lr,
                                           self._params.l_w,
                                           self._params.l_b,
                                           self._seed)

    @property
    def name(self):
        return "KaHFMEmbeddings" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._transactions_per_epoch // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._transactions_per_epoch, self._batch_size):
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
            predictions = self._model.predict_batch(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test
