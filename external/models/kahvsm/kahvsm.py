"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import numpy as np
from tqdm import tqdm
import scipy.sparse as sp

from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from .kahvsm_model import KaHVSMModel
from .IR_feature_weighting import TF_IDF, okapi_BM_25, UI
from .kahvsm_dot import VSM
from elliot.recommender.recommender_utils_mixin import RecMixin


class KaHVSM(RecMixin, BaseRecommenderModel):
    r"""
    Knowledge-aware Hybrid Factorization Machines (Tensorflow Embedding-based Variant)

    Vito Walter Anelli and Tommaso Di Noia and Eugenio Di Sciascio and Azzurra Ragone and Joseph Trotta
    "How to Make Latent Factors Interpretable by Feeding Factorization Machines with Knowledge Graphs", ISWC 2019 Best student Research Paper
    For further details, please refer to the `paper <https://doi.org/10.1007/978-3-030-30793-6_3>`_

    Vito Walter Anelli and Tommaso Di Noia and Eugenio Di Sciascio and Azzurra Ragone and Joseph Trotta
    "Semantic Interpretation of Top-N Recommendations", IEEE TKDE 2020
    For further details, please refer to the `paper <https://doi.org/10.1109/TKDE.2020.3010215>`_

    Args:
        lr: learning rate (default: 0.0001)
        l_w: Weight regularization (default: 0.005)
        l_b: Bias regularization (default: 0)

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        KaHFMEmbeddings:
          meta:
            hyper_max_evals: 20
            hyper_opt_alg: tpe
            validation_rate: 1
            verbose: True
            save_weights: True
            save_recs: True
            validation_metric: nDCG@10
          epochs: 100
          batch_size: -1
          lr: 0.0001
          l_w: 0.005
          l_b: 0

    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a BPR-MF instance.
        (see https://arxiv.org/pdf/1205.2618 for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w, l_b]: regularization,
                                      lr: learning rate}
        """
        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0001, None, None),
            ("_l_w", "l_w", "l_w", 0.005, None, None),
            ("_l_b", "l_b", "l_b", 0, None, None),
            ("_feature_weighting", "feature_weighting", "rel", "TFIDF", None, None),
            ("_k1", "k1", "k1", 1.2, None, None),
            ("_b", "b", "b", 0.75, None, None),
            ("_num_neighbors", "neighbors", "nn", 40, int, None),
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_implicit", "implicit", "bin", False, None, None),
            ("_loader", "loader", "load", "ChainedKG", None, None),
        ]
        self.autoset_params()

        self._side = getattr(self._data.side_information, self._loader, None)

        self._sampler = cs.Sampler(self._data.i_train_dict)

        if self._feature_weighting in ["TFIDF", "BM25"]:
            self._i_feature_dict = {i_item: [self._side.public_features[feature] for feature
                                             in self._side.feature_map[item]] for item, i_item
                                    in self._data.public_items.items()}
            self._sp_i_features = self.build_feature_sparse()

        if self._feature_weighting == "TFIDF":
            self._item_factors_b = TF_IDF(self._sp_i_features)
        elif self._feature_weighting == "BM25":
            self._item_factors_b = okapi_BM_25(self._sp_i_features, K1=self._k1, B=self._b)
        elif self._feature_weighting == "URM":
            self._item_factors_b = UI(self._data.sp_i_train.T)
        else:
            raise ValueError("feature-weighting option not recognized")

        self._user_factors_b = self._data.sp_i_train @ self._item_factors_b

        self._user_factors_b = self._user_factors_b.tocoo()
        bintrain = self._data.sp_i_train.tocoo()

        b = np.bincount(bintrain.row)

        self._user_factors_b.data = self._user_factors_b.data / b[self._user_factors_b.row]

        self._user_factors_b = self._user_factors_b.tocsr()

        if self._batch_size < 1:
            self._batch_size = self._num_users

        self._transactions_per_epoch = self._data.transactions

        self._model = KaHVSMModel(self._user_factors_b.toarray(),
                                           self._item_factors_b.toarray(),
                                           self._params.lr,
                                           self._params.l_w,
                                           self._params.l_b,
                                           self._seed)

    @property
    def name(self):
        return "KaHVSM" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            if self._feature_weighting in ["TFIDF", "BM25"]:
                self.print_info(13, 2858)
            loss = 0
            steps = 0
            with tqdm(total=int(self._transactions_per_epoch // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._transactions_per_epoch, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss.numpy()/(it + 1))
            print()

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._vsm.get_user_recs(u, mask, k) for u in self._data.train_dict.keys()}

    def get_recommendations(self, k: int = 10):
        # user_matrix = self._model.get_user_features()
        item_matrix = self._model.get_item_features().numpy()
        a = self._data.sp_i_train @ self._model.get_item_features().numpy()
        b = np.array(self._data.sp_i_train.sum(axis=1)).squeeze()
        user_matrix = (a.T / b).T
        self._vsm = VSM(data=self._data, user_matrix=user_matrix, item_matrix=item_matrix)

        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def build_feature_sparse(self):

        rows_cols = [(i, f) for i, features in self._i_feature_dict.items() for f in features]
        rows = [u for u, _ in rows_cols]
        cols = [i for _, i in rows_cols]
        data = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                             shape=(self._num_items, len(self._side.public_features)))
        return data

    def print_info(self, user, item):
        iidx = self._data.public_items[item]
        uidx = self._data.public_users[user]
        print(f"Analyzing user {user}")
        print(f"History:")
        for i, v in self._data.train_dict[user].items():
            print(f"Item [{self._side.item_names[i]}] -> {v}")
        print(f"Analyzing item {item}:\t{self._side.item_names[item]}")

        #user
        top_10_indexed_features = np.argsort(- self._model.user_embedding(uidx).numpy())[:30]
        top_10_features = [self._side.private_features[f] for f in top_10_indexed_features]

        for p, f in enumerate(top_10_features):
            fidx = top_10_indexed_features[p]
            print(f"User Feature {self._side.feature_names[f]} -> {self._model.user_embedding.weights[0][uidx, fidx]}")

        #item
        top_10_indexed_features = np.argsort(- self._model.item_embedding(iidx).numpy())[:10]
        top_10_features = [self._side.private_features[f] for f in top_10_indexed_features]

        for p, f in enumerate(top_10_features):
            fidx = top_10_indexed_features[p]
            print(f"Feature [{self._side.feature_names[f]}] -> {self._model.item_embedding.weights[0][iidx,fidx]}")

        #interation
        dot_unpack = self._model.user_embedding(uidx).numpy() * self._model.item_embedding(iidx).numpy()
        bias = self._model.item_bias_embedding(iidx).numpy()
        print(f"Score: {sum(dot_unpack) + bias} -- Dot: {sum(dot_unpack)} -- Bias: {bias}")
        top_10_indexed_features = np.argsort(- dot_unpack)[:10]
        top_10_features = [self._side.private_features[f] for f in top_10_indexed_features]
        print("Most interacted features:")
        for p, f in enumerate(top_10_features):
            fidx = top_10_indexed_features[p]
            print(f"Feature [{self._side.feature_names[f]}] -> {dot_unpack[fidx]}")
