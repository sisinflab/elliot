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


class MFModel(object):
    def __init__(self, F,
                 data,
                 lr,
                 user_regularization,
                 bias_regularization,
                 positive_item_regularization,
                 negative_item_regularization,
                 *args):
        self._factors = F
        self._users = data.users
        self._items = data.items
        self._private_users = data.private_users
        self._public_users = data.public_users
        self._private_items = data.private_items
        self._public_items = data.public_items
        self._learning_rate = lr
        self._user_regularization = user_regularization
        self._bias_regularization = bias_regularization
        self._positive_item_regularization = positive_item_regularization
        self._negative_item_regularization = negative_item_regularization

        self.initialize(*args)

    def initialize(self, loc: float = 0, scale: float = 0.1):
        """
        This function initialize the data model
        :param loc:
        :param scale:
        :return:
        """
        # self._users = list(self._ratings.keys())
        # self._items = list({k for a in self._ratings.values() for k in a.keys()})
        # self._private_users = {p: u for p, u in enumerate(self._users)}
        # self._public_users = {v: k for k, v in self._private_users.items()}
        # self._private_items = {p: i for p, i in enumerate(self._items)}
        # self._public_items = {v: k for k, v in self._private_items.items()}

        self._global_bias = 0

        "same parameters as np.randn"
        self._user_bias = np.zeros(len(self._users))
        self._item_bias = np.zeros(len(self._items))
        self._user_factors = \
            np.random.normal(loc=loc, scale=scale, size=(len(self._users), self._factors))
        self._item_factors = \
            np.random.normal(loc=loc, scale=scale, size=(len(self._items), self._factors))

    @property
    def name(self):
        return "MF"

    def predict(self, user, item):
        return self._global_bias + self._item_bias[self._public_items[item]] \
               + self._user_factors[self._public_users[user]] @ self._item_factors[self._public_items[item]]

    def indexed_predict(self, user, item):
        return self._global_bias + self._item_bias[item] \
               + self._user_factors[user] @ self._item_factors[item]

    # def get_user_recs(self, user, mask, k):
    #     arr = self._item_bias + self._item_factors @ self._user_factors[self._public_users[user]]
    #     local_k = min(k, len(self._ratings[user].keys()) + k)
    #     top_k = arr.argsort()[-(local_k):][::-1]
    #     top_k_2 = [(self._private_items[i], arr[i]) for p, i in enumerate(top_k)
    #                if (self._private_items[i] not in self._ratings[user].keys())]
    #     top_k_2 = top_k_2[:k]
    #     return top_k_2

    def get_user_predictions(self, user_id, mask, top_k=10):
        user_id = self._public_users.get(user_id)
        b = self._item_bias + self._user_factors[user_id] @ self._item_factors.T
        a = mask[user_id]
        b[~a] = -np.inf
        indices, values = zip(*[(self._private_items.get(u_list[0]), u_list[1])
                              for u_list in enumerate(b.data)])

        indices = np.array(indices)
        values = np.array(values)
        local_k = min(top_k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]

    # def get_user_recs_argpartition(self, user: int, k: int):
    #     user_items = self._ratings[user].keys()
    #     local_k = min(k, len(user_items)+k)
    #     predictions = self._item_bias +  self._item_factors  @ self._user_factors[self._public_users[user]]
    #     partially_ordered_preds_indices = np.argpartition(predictions, -local_k)[-local_k:]
    #     partially_ordered_preds_values = predictions[partially_ordered_preds_indices]
    #     partially_ordered_preds_ids = [self._private_items[x] for x in partially_ordered_preds_indices]
    #
    #     top_k = partially_ordered_preds_values.argsort()[::-1]
    #     top_k_2 = [(partially_ordered_preds_ids[i], partially_ordered_preds_values[i]) for p, i in enumerate(top_k)
    #                if (partially_ordered_preds_ids[i] not in user_items)]
    #     top_k_2 = top_k_2[:k]
    #     return top_k_2

    def train_step(self, batch, **kwargs):
        for u, i, j in zip(*batch):
            self.update_factors(u[0], i[0], j[0])

    def update_factors(self, ui: int, ii: int, ji: int):
        user_factors = self._user_factors[ui]
        item_factors_i = self._item_factors[ii]
        item_factors_j = self._item_factors[ji]
        item_bias_i = self._item_bias[ii]
        item_bias_j = self._item_bias[ji]

        z = 1/(1 + np.exp(self.indexed_predict(ui, ii)-self.indexed_predict(ui, ji)))
        # update bias i
        d_bi = (z - self._bias_regularization*item_bias_i)
        self._item_bias[ii] = item_bias_i + (self._learning_rate * d_bi)
        # self.set_item_bias(i, item_bias_i + (self._learning_rate * d_bi))

        # update bias j
        d_bj = (-z - self._bias_regularization*item_bias_j)
        self._item_bias[ji] = item_bias_j + (self._learning_rate * d_bj)
        # self.set_item_bias(j, item_bias_j + (self._learning_rate * d_bj))

        # update user factors
        d_u = ((item_factors_i - item_factors_j)*z - self._user_regularization*user_factors)
        self._user_factors[ui] = user_factors + (self._learning_rate * d_u)
        # self.set_user_factors(u, user_factors + (self._learning_rate * d_u))

        # update item i factors
        d_i = (user_factors*z - self._positive_item_regularization*item_factors_i)
        self._item_factors[ii] = item_factors_i + (self._learning_rate * d_i)
        # self.set_item_factors(i, item_factors_i + (self._learning_rate * d_i))

        # update item j factors
        d_j = (-user_factors*z - self._negative_item_regularization*item_factors_j)
        self._item_factors[ji] = item_factors_j + (self._learning_rate * d_j)
        # self.set_item_factors(j, item_factors_j + (self._learning_rate * d_j))

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

    # def get_user_bias(self, user: int):
    #
    #     return self._user_bias[self._public_users[user]]
    #
    # def get_item_bias(self, item: int):
    #
    #     return self._item_bias[self._public_items[item]]
    #
    # def get_user_factors(self, user: int):
    #
    #     return self._user_factors[self._public_users[user]]
    #
    # def get_item_factors(self, item: int):
    #
    #     return self._item_factors[self._public_items[item]]
    #
    # def set_user_bias(self, user: int, v: float):
    #
    #     self._user_bias[self._public_users[user]] = v
    #
    # def set_item_bias(self, item: int, v: float):
    #
    #     self._item_bias[self._public_items[item]] = v
    #
    # def set_user_factors(self, user: int, v: float):
    #
    #     self._user_factors[self._public_users[user]] = v
    #
    # def set_item_factors(self, item: int, v: float):
    #
    #     self._item_factors[self._public_items[item]] = v