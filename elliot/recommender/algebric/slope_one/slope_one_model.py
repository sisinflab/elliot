
"""

Lemire, Daniel, and Anna Maclachlan. "Slope one predictors for online rating-based collaborative filtering."
Proceedings of the 2005 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics
"""
import pickle

import numpy as np


class SlopeOneModel:
    def __init__(self, data):
        self._data = data
        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._i_train = self._data.i_train_dict

    def initialize(self):
        freq = np.empty((self._num_items, self._num_items))
        dev = np.empty((self._num_items, self._num_items))

        # Computation of freq and dev arrays.
        for u, u_ratings in self._i_train.items():
            for i, r_ui in u_ratings.items():
                for j, r_uj in u_ratings.items():
                    freq[i, j] += 1
                    dev[i, j] += r_ui - r_uj

        for i in range(self._num_items):
            dev[i, i] = 0
            for j in range(i + 1, self._num_items):
                dev[i, j] = dev[i, j]/freq[i, j] if freq[i, j] != 0 else 0
                dev[j, i] = -dev[i, j]

        self.freq = freq
        self.dev = dev

        # mean ratings of all users: mu_u
        self.user_mean = [np.mean([r for (_, r) in self._i_train[u].items()]) for u in range(self._num_users)]

    def predict(self, user, item):
        Ri = [j for (j, _) in self._i_train[user].items() if self.freq[item, j] > 0]
        pred = self.user_mean[user]
        if Ri:
            pred += sum(self.dev[item, j] for j in Ri) / len(Ri)
        return pred

    def get_user_recs(self, u, mask, k=100):
        uidx = self._data.public_users[u]
        user_mask = mask[uidx]
        # user_items = self._data.train_dict[u].keys()
        # indexed_user_items = [self._data.public_items[i] for i in user_items]
        predictions = {self._data.private_items[iidx]: self.predict(uidx, iidx) for iidx in range(self._num_items) if user_mask[iidx]}

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
        saving_dict['freq'] = self.freq
        saving_dict['dev'] = self.dev
        saving_dict['user_mean'] = self.user_mean
        return saving_dict

    def set_model_state(self, saving_dict):
        self.freq = saving_dict['freq']
        self.dev = saving_dict['dev']
        self.user_mean = saving_dict['user_mean']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
