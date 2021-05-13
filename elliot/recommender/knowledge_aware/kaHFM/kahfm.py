import time
import numpy as np
import pickle
import typing as t
from tqdm import tqdm


from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation
from elliot.recommender.knowledge_aware.kaHFM.tfidf_utils import TFIDF
from elliot.recommender.knowledge_aware.kaHFM.kahfm_model import KAHFMModel
from elliot.recommender.base_recommender_model import init_charger


class KaHFM(RecMixin, BaseRecommenderModel):
    r"""
    Knowledge-aware Hybrid Factorization Machines

    Vito Walter Anelli and Tommaso Di Noia and Eugenio Di Sciascio and Azzurra Ragone and Joseph Trotta
    "How to Make Latent Factors Interpretable by Feeding Factorization Machines with Knowledge Graphs", ISWC 2019 Best student Research Paper
    For further details, please refer to the `paper <https://doi.org/10.1007/978-3-030-30793-6_3>`_

    Vito Walter Anelli and Tommaso Di Noia and Eugenio Di Sciascio and Azzurra Ragone and Joseph Trotta
    "Semantic Interpretation of Top-N Recommendations", IEEE TKDE 2020
    For further details, please refer to the `paper <https://doi.org/10.1109/TKDE.2020.3010215>`_

    Args:
        lr: learning rate (default: 0.05)
        bias_regularization: Bias regularization (default: 0)
        user_regularization: User regularization (default: 0.0025)
        positive_item_regularization: regularization for positive (experienced) items (default: 0.0025)
        negative_item_regularization: regularization for unknown items (default: 0.00025)
        update_negative_item_factors: Boolean to update negative item factors (default: True)
        update_users: Boolean to update user factors (default: True)
        update_items: Boolean to update item factors (default: True)
        update_bias: Boolean to update bias value (default: True)

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        KaHFM:
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
          lr: 0.05
          bias_regularization: 0
          user_regularization: 0.0025
          positive_item_regularization: 0.0025
          negative_item_regularization: 0.00025
          update_negative_item_factors: True
          update_users: True
          update_items: True
          update_bias: True

    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.05, None, None),
            ("_bias_regularization", "bias_regularization", "b_reg", 0, None, None),
            ("_user_regularization", "user_regularization", "u_reg", 0.0025,
             None, None),
            ("_positive_item_regularization", "positive_item_regularization", "pos_i_reg", 0.0025,
             None, None),
            ("_negative_item_regularization", "negative_item_regularization", "neg_it_reg", 0.00025,
             None, None),
            # ("_update_negative_item_factors", "update_negative_item_factors", "update_neg_item_factors",True,
            #  None, None),
            # ("_update_users", "update_users", "update_users", True, None, None),
            # ("_update_items", "update_items", "update_items", True, None, None),
            # ("_update_bias", "update_bias", "update_bias", True, None, None),
            ("_loader", "loader", "load", "ChainedKG", None, None),
        ]
        self.autoset_params()
        self._sample_negative_items_empirically = True

        self._ratings = self._data.train_dict

        self._side = getattr(self._data.side_information, self._loader, None)

        self._tfidf_obj = TFIDF(self._side.feature_map)
        self._tfidf = self._tfidf_obj.tfidf()
        self._user_profiles = self._tfidf_obj.get_profiles(self._ratings)

        self._model = KAHFMModel(self._data,
                                 self._side,
                                 self._tfidf,
                                 self._user_profiles,
                                 self._learning_rate,
                                 self._user_regularization,
                                 self._bias_regularization,
                                 self._positive_item_regularization,
                                 self._negative_item_regularization)
        self._embed_k = self._model.get_factors()
        self._sampler = cs.Sampler(self._data.i_train_dict)
        self._batch_size = 10000

    def get_recommendations(self, k: int = 10):
        self._model.prepare_predictions()

        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_predictions(u, mask, k) for u in self._ratings.keys()}

    # def get_recommendations(self, k: int = 100):
    #     return {u: self._model.get_user_recs(u, k) for u in self._ratings.keys()}

    # def predict(self, u: int, i: int):
    #     """
    #     Get prediction on the user item pair.
    #
    #     Returns:
    #         A single float vaue.
    #     """
    #     return self._model.predict(u, i)

    @property
    def name(self):
        return "KaHFM" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    # def train_step(self):
    #     if self._restore:
    #         return self.restore_weights()
    #
    #     start_it = time.perf_counter()
    #     print()
    #     print("Sampling...")
    #     samples = self._sampler.step(self._data.transactions)
    #     start = time.perf_counter()
    #     print(f"Sampled in {round(start-start_it, 2)} seconds")
    #     start = time.perf_counter()
    #     print("Computing..")
    #     for u, i, j in samples:
    #         self.update_factors(u, i, j)
    #     t2 = time.perf_counter()
    #     print(f"Computed and updated in {round(t2-start, 2)} seconds")

    def train(self):
        if self._restore:
            return self.restore_weights()

        print(f"Transactions: {self._data.transactions}")

        for it in self.iterate(self._epochs):
            print(f"\n********** Iteration: {it + 1}")
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    self._model.train_step(batch)
                    t.update()

            self.evaluate(it)

    # def update_factors(self, u: int, i: int, j: int):
    #     user_factors = self._model.get_user_factors(u)
    #     item_factors_i = self._model.get_item_factors(i)
    #     item_factors_j = self._model.get_item_factors(j)
    #     item_bias_i = self._model.get_item_bias(i)
    #     item_bias_j = self._model.get_item_bias(j)
    #
    #     z = 1 / (1 + np.exp(self.predict(u, i) - self.predict(u, j)))
    #     # update bias i
    #     d_bi = (z - self._bias_regularization * item_bias_i)
    #     self._model.set_item_bias(i, item_bias_i + (self._learning_rate * d_bi))
    #
    #     # update bias j
    #     d_bj = (-z - self._bias_regularization * item_bias_j)
    #     self._model.set_item_bias(j, item_bias_j + (self._learning_rate * d_bj))
    #
    #     # update user factors
    #     d_u = ((item_factors_i - item_factors_j) * z - self._user_regularization * user_factors)
    #     self._model.set_user_factors(u, user_factors + (self._learning_rate * d_u))
    #
    #     # update item i factors
    #     d_i = (user_factors * z - self._positive_item_regularization * item_factors_i)
    #     self._model.set_item_factors(i, item_factors_i + (self._learning_rate * d_i))
    #
    #     # update item j factors
    #     d_j = (-user_factors * z - self._negative_item_regularization * item_factors_j)
    #     self._model.set_item_factors(j, item_factors_j + (self._learning_rate * d_j))

