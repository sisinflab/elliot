"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle

import numpy as np


class BPRSlimModel(object):
    def __init__(self,
                 data, num_users, num_items, lr, lj_reg, li_reg, sampler, random_seed=42):

        self._data = data
        self._num_users = num_users
        self._num_items = num_items
        self._sp_i_train_ratings = self._data.sp_i_train_ratings
        self._lr = lr
        self._lj_reg = lj_reg
        self._li_reg = li_reg
        self._sampler = sampler

        self._random_seed = random_seed
        self._random_state = np.random.RandomState(self._random_seed)

        self._mask_indices = np.array(self._sp_i_train_ratings.indices, dtype=np.int32)
        self._mask_indptr = np.array(self._sp_i_train_ratings.indptr, dtype=np.int32)

        self._s_dense = np.empty((self._num_items, self._num_items), np.double)

    def train_step(self, batch):
        u, i, j = batch
        x_uij = 0.0

        # The difference is computed on the user_seen items
        index = 0

        seen_items_start_pos = self._mask_indptr[u]
        seen_items_end_pos = self._mask_indptr[u + 1]

        while index < seen_items_end_pos - seen_items_start_pos:
            seenItem = self._mask_indices[seen_items_start_pos + index]
            index += 1

            x_uij += self._s_dense[i, seenItem] - self._s_dense[j, seenItem]

        gradient = 1 / (1 + np.exp(x_uij))
        loss = np.sum(x_uij) ** 2

        index = 0
        while index < seen_items_end_pos - seen_items_start_pos:

            seenItem = self._mask_indices[seen_items_start_pos + index]
            index += 1

            if seenItem != i:
                self._s_dense[i, seenItem] += self._lr * (gradient - self._li_reg * self._s_dense[i, seenItem])

            if seenItem != j:
                self._s_dense[j, seenItem] -= self._lr * (gradient - self._lj_reg * self._s_dense[j, seenItem])

        return loss

    def predict(self, u, i):
        x_ui = 0.0

        # The difference is computed on the user_seen items
        index = 0

        seen_items_start_pos = self._mask_indptr[u]
        seen_items_end_pos = self._mask_indptr[u + 1]

        while index < seen_items_end_pos - seen_items_start_pos:
            seenItem = self._mask_indices[seen_items_start_pos + index]
            index += 1

            x_ui += self._s_dense[i, seenItem]

        return x_ui

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
        saving_dict['_s_dense'] = self._s_dense
        return saving_dict

    def set_model_state(self, saving_dict):
        self._s_dense = saving_dict['_s_dense']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
