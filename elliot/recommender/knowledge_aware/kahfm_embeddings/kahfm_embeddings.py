"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import numpy as np
from tqdm import tqdm

from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.knowledge_aware.kaHFM_batch.tfidf_utils import TFIDF
from elliot.recommender.knowledge_aware.kahfm_embeddings.kahfm_embeddings_model import KaHFMEmbeddingsModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation


class KaHFMEmbeddings(RecMixin, BaseRecommenderModel):
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
            ("_loader", "loader", "load", "ChainedKG", None, None),
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict

        self._side = getattr(self._data.side_information, self._loader, None)

        self._sampler = cs.Sampler(self._data.i_train_dict)

        self._tfidf_obj = TFIDF(self._side.feature_map)
        self._tfidf = self._tfidf_obj.tfidf()
        self._user_profiles = self._tfidf_obj.get_profiles(self._ratings)

        self._user_factors = \
            np.zeros(shape=(len(self._data.users), len(self._side.features)))
        self._item_factors = \
            np.zeros(shape=(len(self._data.items), len(self._side.features)))

        for i, f_dict in self._tfidf.items():
            if i in self._data.items:
                for f, v in f_dict.items():
                    self._item_factors[self._data.public_items[i]][self._side.public_features[f]] = v

        for u, f_dict in self._user_profiles.items():
            for f, v in f_dict.items():
                self._user_factors[self._data.public_users[u]][self._side.public_features[f]] = v

        if self._batch_size < 1:
            self._batch_size = self._num_users


        # self._factors = self._data.factors

        self._transactions_per_epoch = self._data.transactions

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
