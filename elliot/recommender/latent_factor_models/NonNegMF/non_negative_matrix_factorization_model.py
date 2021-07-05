"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle

import numpy as np


class NonNegMFModel(object):
    def __init__(self,
                 data,
                 num_users,
                 num_items,
                 global_mean,
                 embed_mf_size,
                 lambda_weights,
                 learning_rate=0.01,
                 random_seed=42):

        self._data = data
        self._i_train = self._data.i_train_dict
        self._learning_rate = learning_rate
        self._random_seed = random_seed
        self._random_state = np.random.RandomState(self._random_seed)
        self._num_users = num_users
        self._num_items = num_items
        self._global_mean = global_mean
        self._embed_mf_size = embed_mf_size
        self._lambda_weights = lambda_weights

        self._user_bias = np.empty(self._num_users, np.double)
        self._item_bias = np.empty(self._num_items, np.double)
        self._user_embeddings = self._random_state.normal(size=(self._num_users, self._embed_mf_size))
        self._item_embeddings = self._random_state.normal(size=(self._num_items, self._embed_mf_size))

    def train_step(self):

        # (re)initialize nums and denominators to zero
        user_num = np.empty((self._num_users, self._embed_mf_size))
        user_denom = np.empty((self._num_users, self._embed_mf_size))
        item_num = np.empty((self._num_items, self._embed_mf_size))
        item_denom = np.empty((self._num_items, self._embed_mf_size))

        # Compute numerators and denominators for users and items factors
        for u, u_ratings in self._i_train.items():
            for i, r_ui in u_ratings.items():

                # compute current estimation and error
                dot = 0  # <q_i, p_u>
                for f in range(self._embed_mf_size):
                    dot += self._item_embeddings[i, f] * self._user_embeddings[u, f]
                est = self._global_mean + self._user_bias[u] + self._item_bias[i] + dot
                err = r_ui - est

                # update biases
                self._user_bias[u] += self._learning_rate * (err - self._lambda_weights * self._user_bias[u])
                self._item_bias[i] += self._learning_rate * (err - self._lambda_weights * self._item_bias[i])

                # compute numerators and denominators
                for f in range(self._embed_mf_size):
                    user_num[u, f] += self._item_embeddings[i, f] * r_ui
                    user_denom[u, f] += self._item_embeddings[i, f] * est
                    item_num[i, f] += self._user_embeddings[u, f] * r_ui
                    item_denom[i, f] += self._user_embeddings[u, f] * est

        # Update user factors
        for u, u_ratings in self._i_train.items():
            n_ratings = len(u_ratings)
            for f in range(self._embed_mf_size):
                user_denom[u, f] += n_ratings * self._lambda_weights * self._user_embeddings[u, f]
                self._user_embeddings[u, f] *= user_num[u, f] / user_denom[u, f]

        # Update item factors
        for i in range(self._num_items):
            n_ratings = self._data.sp_i_train.getcol(i).nnz
            for f in range(self._embed_mf_size):
                item_denom[i, f] += n_ratings * self._lambda_weights * self._item_embeddings[i, f]
                self._item_embeddings[i, f] *= item_num[i, f] / item_denom[i, f]

    def predict(self, user, item):
        return self._user_embeddings[self._data.public_users[user], :].dot(
            self._item_embeddings[self._data.public_items[item], :]) + self._item_bias[self._data.public_items[item]] + self._user_bias[self._data.public_users[user]] + self._global_mean

    def get_user_recs(self, user, mask, k=100):
        user_mask = mask[self._data.public_users[user]]
        predictions = {i: self.predict(user, i) for i in self._data.items if user_mask[self._data.public_items[i]]}

        indices, values = zip(*predictions.items())
        indices = np.array(indices)
        values = np.array(values)
        local_k = min(k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]

    def get_model_state(self):
        saving_dict = {}
        saving_dict['_user_bias'] = self._user_bias
        saving_dict['_item_bias'] = self._item_bias
        saving_dict['_user_embeddings'] = self._user_embeddings
        saving_dict['_item_embeddings'] = self._item_embeddings
        return saving_dict

    def set_model_state(self, saving_dict):
        self._user_bias = saving_dict['_user_bias']
        self._item_bias = saving_dict['_item_bias']
        self._user_embeddings = saving_dict['_user_embeddings']
        self._item_embeddings = saving_dict['_item_embeddings']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
