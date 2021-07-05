"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle

import numpy as np
from scipy import sparse as sp
from sklearn.utils.extmath import randomized_svd


class PureSVDModel(object):
    """
    Simple Matrix Factorization class
    """

    def __init__(self, factors, data, random_seed):

        self._data = data
        self._private_users = data.private_users
        self._public_users = data.public_users
        self._private_items = data.private_items
        self._public_items = data.public_items
        self.factors = factors
        self.random_seed = random_seed
        self.train_dict = self._data.train_dict
        self.user_num, self.item_num = self._data.num_users, self._data.num_items

        self.user_vec, self.item_vec = None, None

    def train_step(self):
        U, sigma, Vt = randomized_svd(self._data.sp_i_train,
                                      n_components=self.factors,
                                      random_state=self.random_seed)
        s_Vt = sp.diags(sigma) * Vt

        self.user_vec = U
        self.item_vec = s_Vt.T

    def predict(self, user, item):
        return self.user_vec[self._data.public_users[user], :].dot(self.item_vec[self._data.public_items[item], :])

    def get_user_recs(self, user_id, mask, top_k=100):
        # user_items = self._data.train_dict[user].keys()
        # k = min(k, self._data.num_items - len(user_items))
        # predictions = {i: self.predict(user, i) for i in self._data.items if i not in user_items}
        # indices, values = zip(*predictions.items())
        # indices = np.array(indices)
        # values = np.array(values)
        # local_k = min(k, len(values))
        # partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        # real_values = values[partially_ordered_preds_indices]
        # real_indices = indices[partially_ordered_preds_indices]
        # local_top_k = real_values.argsort()[::-1]
        # return [(real_indices[item], real_values[item]) for item in local_top_k]
        user_id = self._public_users.get(user_id)
        b = self.user_vec[user_id] @ self.item_vec.T
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

    def get_model_state(self):
        saving_dict = {}
        saving_dict['user_vec'] = self.user_vec
        saving_dict['item_vec'] = self.item_vec
        return saving_dict

    def set_model_state(self, saving_dict):
        self.user_vec = saving_dict['user_vec']
        self.item_vec = saving_dict['item_vec']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
