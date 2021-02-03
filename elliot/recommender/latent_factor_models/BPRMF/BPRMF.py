"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import time
import numpy as np
import pickle

from dataset.samplers import pairwise_sampler as ps
from evaluation.evaluator import Evaluator
from recommender.recommender_utils_mixin import RecMixin
from utils.folder import build_model_folder
from utils.write import store_recommendation

from recommender.base_recommender_model import BaseRecommenderModel

np.random.seed(42)


class MF(object):
    """
    Simple Matrix Factorization class
    """

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
        top_k = arr.argsort()[-(len(self._ratings[user].keys()) + k):][::-1]
        top_k_2 = [(self._private_items[i], arr[i]) for p, i in enumerate(top_k)
                   if (self._private_items[i] not in self._ratings[user].keys())]
        top_k_2 = top_k_2[:k]
        return top_k_2

    def get_user_recs_argpartition(self, user: int, k: int):
        user_items = self._ratings[user].keys()
        safety_k = len(user_items)+k
        predictions = self._item_bias +  self._item_factors  @ self._user_factors[self._public_users[user]]
        partially_ordered_preds_indices = np.argpartition(predictions, -safety_k)[-safety_k:]
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

    def __init__(self, data, config, params, *args, **kwargs):
        super().__init__(data, config, params, *args, **kwargs)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random
        self._sample_negative_items_empirically = True

        self._factors = self._params.factors
        self._learning_rate = self._params.lr
        self._bias_regularization = self._params.bias_regularization
        self._user_regularization = self._params.user_regularization
        self._positive_item_regularization = self._params.positive_item_regularization
        self._negative_item_regularization = self._params.negative_item_regularization
        self._update_negative_item_factors = self._params.update_negative_item_factors
        self._update_users = self._params.update_users
        self._update_items = self._params.update_items
        self._update_bias = self._params.update_bias

        self._ratings = self._data.train_dict
        self._datamodel = MF(self._factors, self._ratings, self._random)
        self._sampler = ps.Sampler(self._ratings, self._data.users, self._data.items)

        self._iteration = 0

        self.evaluator = Evaluator(self._data, self._params)

        self._params.name = self.name

        build_model_folder(self._config.path_output_rec_weight, self.name)
        self._saving_filepath = f'{self._config.path_output_rec_weight}{self.name}/best-weights-{self.name}'

    def get_recommendations(self, k: int = 100):
        return {u: self._datamodel.get_user_recs(u, k) for u in self._ratings.keys()}

    def predict(self, u: int, i: int):
        """
        Get prediction on the user item pair.

        Returns:
            A single float vaue.
        """
        return self._datamodel.predict(u, i)

    @property
    def name(self):
        return "SequentialBPRMF" \
               + self._datamodel.name \
               + "_lr:" + str(self._learning_rate) \
               + "-e:" + str(self._epochs) \
               + "-factors:" + str(self._factors) \
               + "-br:" + str(self._bias_regularization) \
               + "-ur:" + str(self._user_regularization) \
               + "-pir:" + str(self._positive_item_regularization) \
               + "-nir:" + str(self._negative_item_regularization)

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
        print(f"Transactions: {self._data.transactions}")
        best_metric_value = -np.inf
        for it in range(self._epochs):
            self.restore_weights(it)
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
                            pickle.dump(self._datamodel.get_model_state(), f)
                    if self._save_recs:
                        store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

    def update_factors(self, u: int, i: int, j: int):
        user_factors = self._datamodel.get_user_factors(u)
        item_factors_i = self._datamodel.get_item_factors(i)
        item_factors_j = self._datamodel.get_item_factors(j)
        item_bias_i = self._datamodel.get_item_bias(i)
        item_bias_j = self._datamodel.get_item_bias(j)

        z = 1/(1 + np.exp(self.predict(u, i)-self.predict(u, j)))
        # update bias i
        d_bi = (z - self._bias_regularization*item_bias_i)
        self._datamodel.set_item_bias(i, item_bias_i + (self._learning_rate * d_bi))

        # update bias j
        d_bj = (-z - self._bias_regularization*item_bias_j)
        self._datamodel.set_item_bias(j, item_bias_j + (self._learning_rate * d_bj))

        # update user factors
        d_u = ((item_factors_i - item_factors_j)*z - self._user_regularization*user_factors)
        self._datamodel.set_user_factors(u, user_factors + (self._learning_rate * d_u))

        # update item i factors
        d_i = (user_factors*z - self._positive_item_regularization*item_factors_i)
        self._datamodel.set_item_factors(i, item_factors_i + (self._learning_rate * d_i))

        # update item j factors
        d_j = (-user_factors*z - self._negative_item_regularization*item_factors_j)
        self._datamodel.set_item_factors(j, item_factors_j + (self._learning_rate * d_j))

    # def get_loss(self):
    #     return -max([r[self._validation_metric] for r in self._results])

    # def get_params(self):
    #     return self._params.__dict__

    # def get_results(self):
    #     val_max = np.argmax([r[self._validation_metric] for r in self._results])
    #     return self._results[val_max]

    # def get_statistical_results(self):
    #     val_max = np.argmax([r[self._validation_metric] for r in self._results])
    #     return self._statistical_results[val_max]

    def restore_weights(self, it):
        if self._restore_epochs == it:
            try:
                with open(self._saving_filepath, "rb") as f:
                    self._datamodel.set_model_state(pickle.load(f))
                print(f"Model correctly Restored at Epoch: {self._restore_epochs}")
                return True
            except Exception as ex:
                print(f"Error in model restoring operation! {ex}")
        return False
