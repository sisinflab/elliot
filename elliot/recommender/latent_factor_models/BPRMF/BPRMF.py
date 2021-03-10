"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle
import time

import numpy as np

from elliot.dataset.samplers import pairwise_sampler as ps
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

np.random.seed(42)


class MF(object):
    def __init__(self, F, ratings, random, *args):
        self._factors = F
        self._ratings = ratings
        self._random = random
        self.initialize(*args)

    def initialize(self, loc: float = 0, scale: float = 0.1):
        """
        This function initialize the data model
        :param loc:
        :param scale:
        :return:
        """
        self._users = list(self._ratings.keys())
        self._items = list({k for a in self._ratings.values() for k in a.keys()})
        self._private_users = {p: u for p, u in enumerate(self._users)}
        self._public_users = {v: k for k, v in self._private_users.items()}
        self._private_items = {p: i for p, i in enumerate(self._items)}
        self._public_items = {v: k for k, v in self._private_items.items()}

        self._global_bias = 0

        "same parameters as np.randn"
        self._user_bias = np.zeros(len(self._users))
        self._item_bias = np.zeros(len(self._items))
        self._user_factors = \
            self._random.normal(loc=loc, scale=scale, size=(len(self._users), self._factors))
        self._item_factors = \
            self._random.normal(loc=loc, scale=scale, size=(len(self._items), self._factors))
        self._transactions = sum(len(v) for v in self._ratings.values())

    @property
    def name(self):
        return "MF"

    def get_transactions(self):
        return self._transactions

    def predict(self, user, item):
        return self._global_bias + self._item_bias[self._public_items[item]] \
               + self._user_factors[self._public_users[user]] @ self._item_factors[self._public_items[item]]

    def get_user_recs(self, user, k):
        arr = self._item_bias + self._item_factors @ self._user_factors[self._public_users[user]]
        local_k = min(k, len(self._ratings[user].keys()) + k)
        top_k = arr.argsort()[-(local_k):][::-1]
        top_k_2 = [(self._private_items[i], arr[i]) for p, i in enumerate(top_k)
                   if (self._private_items[i] not in self._ratings[user].keys())]
        top_k_2 = top_k_2[:k]
        return top_k_2

    def get_user_recs_argpartition(self, user: int, k: int):
        user_items = self._ratings[user].keys()
        local_k = min(k, len(user_items)+k)
        predictions = self._item_bias +  self._item_factors  @ self._user_factors[self._public_users[user]]
        partially_ordered_preds_indices = np.argpartition(predictions, -local_k)[-local_k:]
        partially_ordered_preds_values = predictions[partially_ordered_preds_indices]
        partially_ordered_preds_ids = [self._private_items[x] for x in partially_ordered_preds_indices]

        top_k = partially_ordered_preds_values.argsort()[::-1]
        top_k_2 = [(partially_ordered_preds_ids[i], partially_ordered_preds_values[i]) for p, i in enumerate(top_k)
                   if (partially_ordered_preds_ids[i] not in user_items)]
        top_k_2 = top_k_2[:k]
        return top_k_2

    def get_model_state(self):
        saving_dict = {}
        saving_dict['_user_bias'] = self._user_bias
        saving_dict['_item_bias'] = self._item_bias
        saving_dict['_user_factors'] = self._user_factors
        saving_dict['_item_factors'] = self._item_factors
        return saving_dict

    def set_model_state(self, saving_dict):
        self._user_bias = saving_dict['_user_bias']
        self._item_bias = saving_dict['_item_bias']
        self._user_factors = saving_dict['_user_factors']
        self._item_factors = saving_dict['_item_factors']

    def get_user_bias(self, user: int):

        return self._user_bias[self._public_users[user]]

    def get_item_bias(self, item: int):

        return self._item_bias[self._public_items[item]]

    def get_user_factors(self, user: int):

        return self._user_factors[self._public_users[user]]

    def get_item_factors(self, item: int):

        return self._item_factors[self._public_items[item]]

    def set_user_bias(self, user: int, v: float):

        self._user_bias[self._public_users[user]] = v

    def set_item_bias(self, item: int, v: float):

        self._item_bias[self._public_items[item]] = v

    def set_user_factors(self, user: int, v: float):

        self._user_factors[self._public_users[user]] = v

    def set_item_factors(self, item: int, v: float):

        self._item_factors[self._public_items[item]] = v


class BPRMF(RecMixin, BaseRecommenderModel):
    r"""
    Bayesian Personalized Ranking with Matrix Factorization

    For further details, please refer to the `paper <https://arxiv.org/abs/1205.2618.pdf>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        bias_regularization: Regularization coefficient for the bias
        user_regularization: Regularization coefficient for user latent factors
        positive_item_regularization: Regularization coefficient for positive item latent factors
        negative_item_regularization: Regularization coefficient for negative item latent factors
        update_negative_item_factors:
        update_users:
        update_items:
        update_bias:


    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        BPRMF:
          meta:
            save_recs: True
          epochs: 10
          factors: 10
          lr: 0.001
          bias_regularization: 0
          user_regularization: 0.0025
          positive_item_regularization: 0.0025
          negative_item_regularization: 0.0025
          update_negative_item_factors: True
          update_users: True
          update_items: True
          update_bias: True
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._random = np.random

        self._params_list = [
            ("_factors", "factors", "f", 10, int, None),
            ("_learning_rate", "lr", "lr", 0.05, None, None),
            ("_bias_regularization", "bias_regularization", "bias_reg", 0, None, None),
            ("_user_regularization", "user_regularization", "u_reg", 0.0025,
             None, None),
            ("_positive_item_regularization", "positive_item_regularization", "pos_i_reg", 0.0025,
             None, None),
            ("_negative_item_regularization", "negative_item_regularization", "neg_i_reg", 0.00025,
             None, None),
            ("_update_negative_item_factors", "update_negative_item_factors", "up_neg_i_f", True,
             None, None),
            ("_update_users", "update_users", "up_u", True, None, None),
            ("_update_items", "update_items", "up_i", True, None, None),
            ("_update_bias", "update_bias", "up_b", True, None, None),
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict
        self._model = MF(self._factors, self._ratings, self._random)
        self._sampler = ps.Sampler(self._ratings, self._data.users, self._data.items)

    def get_recommendations(self, k: int = 100):
        return {u: self._model.get_user_recs(u, k) for u in self._ratings.keys()}

    def predict(self, u: int, i: int):
        """
        Get prediction on the user item pair.

        Returns:
            A single float vaue.
        """
        return self._model.predict(u, i)

    @property
    def name(self):
        return "BPRMF" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train_step(self):
        start_it = time.perf_counter()
        print()
        print("Sampling...")
        samples = self._sampler.step(self._data.transactions)
        start = time.perf_counter()
        print(f"Sampled in {round(start-start_it, 2)} seconds")
        start = time.perf_counter()
        print("Computing..")
        for u, i, j in samples:
            self.update_factors(u, i, j)
        t2 = time.perf_counter()
        print(f"Computed and updated in {round(t2-start, 2)} seconds")

    def train(self):
        if self._restore:
            return self.restore_weights()

        print(f"Transactions: {self._data.transactions}")
        best_metric_value = -np.inf
        for it in range(self._epochs):
            print(f"\n********** Iteration: {it + 1}")
            self._iteration = it

            self.train_step()

            if not (it + 1) % self._validation_rate:
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                result_dict = self.evaluator.eval(recs)
                self._results.append(result_dict)

                if self._results[-1][self._validation_k]["val_results"][self._validation_metric] > best_metric_value:
                    print("******************************************")
                    best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                    if self._save_weights:
                        with open(self._saving_filepath, "wb") as f:
                            pickle.dump(self._model.get_model_state(), f)
                    if self._save_recs:
                        store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

    def update_factors(self, u: int, i: int, j: int):
        user_factors = self._model.get_user_factors(u)
        item_factors_i = self._model.get_item_factors(i)
        item_factors_j = self._model.get_item_factors(j)
        item_bias_i = self._model.get_item_bias(i)
        item_bias_j = self._model.get_item_bias(j)

        z = 1/(1 + np.exp(self.predict(u, i)-self.predict(u, j)))
        # update bias i
        d_bi = (z - self._bias_regularization*item_bias_i)
        self._model.set_item_bias(i, item_bias_i + (self._learning_rate * d_bi))

        # update bias j
        d_bj = (-z - self._bias_regularization*item_bias_j)
        self._model.set_item_bias(j, item_bias_j + (self._learning_rate * d_bj))

        # update user factors
        d_u = ((item_factors_i - item_factors_j)*z - self._user_regularization*user_factors)
        self._model.set_user_factors(u, user_factors + (self._learning_rate * d_u))

        # update item i factors
        d_i = (user_factors*z - self._positive_item_regularization*item_factors_i)
        self._model.set_item_factors(i, item_factors_i + (self._learning_rate * d_i))

        # update item j factors
        d_j = (-user_factors*z - self._negative_item_regularization*item_factors_j)
        self._model.set_item_factors(j, item_factors_j + (self._learning_rate * d_j))

    def restore_weights(self):
        try:
            with open(self._saving_filepath, "rb") as f:
                self._model.set_model_state(pickle.load(f))
            print(f"Model correctly Restored")

            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)
            self._results.append(result_dict)

            print("******************************************")
            if self._save_recs:
                store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")
            return True

        except Exception as ex:
            print(f"Error in model restoring operation! {ex}")

        return False
