"""
Module description:

"""

__version__ = '0.1'
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

np.random.seed(42)


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
        self._random = np.random

        self._ratings = self._data.train_dict
        self._sampler = cs.Sampler(self._data.i_train_dict)

        self._tfidf_obj = TFIDF(self._data.side_information_data.feature_map)
        self._tfidf = self._tfidf_obj.tfidf()
        self._user_profiles = self._tfidf_obj.get_profiles(self._ratings)

        self._user_factors = \
            np.zeros(shape=(len(self._data.users), len(self._data.features)))
        self._item_factors = \
            np.zeros(shape=(len(self._data.items), len(self._data.features)))

        for i, f_dict in self._tfidf.items():
            if i in self._data.items:
                for f, v in f_dict.items():
                    self._item_factors[self._data.public_items[i]][self._data.public_features[f]] = v

        for u, f_dict in self._user_profiles.items():
            for f, v in f_dict.items():
                self._user_factors[self._data.public_users[u]][self._data.public_features[f]] = v

        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0001, None, None),
            ("_l_w", "l_w", "l_w", 0.005, None, None),
            ("_l_b", "l_b", "l_b", 0, None, None),
        ]
        self.autoset_params()

        self._factors = self._data.factors

        self._transactions_per_epoch = self._data.transactions

        self._model = KaHFMEmbeddingsModel(self._user_factors,
                                  self._item_factors,
                                  self._params.lr,
                                  self._params.l_w,
                                  self._params.l_b)

    @property
    def name(self):
        return "KaHFMEmbeddings" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        best_metric_value = 0
        for it in range(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._transactions_per_epoch // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._transactions_per_epoch, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            if not (it + 1) % self._validation_rate:
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                result_dict = self.evaluator.eval(recs)
                self._results.append(result_dict)

                print(f'Epoch {(it + 1)}/{self._epochs} loss {loss:.3f}')

                if self._results[-1][self._validation_k]["val_results"][self._validation_metric] > best_metric_value:
                    print("******************************************")
                    best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                    if self._save_weights:
                        self._model.save_weights(self._saving_filepath)
                    if self._save_recs:
                        store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset+self._batch_size, self._num_users)
            predictions = self._model.predict_batch(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, self.get_train_mask(offset, offset_stop), k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
        return predictions_top_k
