import pickle

import numpy as np


class VSM(object):
    """
    Simple VSM class
    """

    def __init__(self, data, user_matrix, item_matrix):
        self._data = data
        self._ratings = data.train_dict
        self._user_matrix = user_matrix
        self._item_matrix = item_matrix

        self._users = self._data.users
        self._items = self._data.items
        self._private_users = self._data.private_users
        self._public_users = self._data.public_users
        self._private_items = self._data.private_items
        self._public_items = self._data.public_items

        self._preds = self._user_matrix @ self._item_matrix.T

    def get_user_recs(self, u, mask, k):
        user_id = self._data.public_users.get(u)
        user_recs = self._preds[user_id]
        user_recs_mask = mask[user_id]
        user_recs[~user_recs_mask] = -np.inf
        indices, values = zip(*[(self._data.private_items.get(u_list[0]), u_list[1])
                              for u_list in enumerate(user_recs)])

        indices = np.array(indices)
        values = np.array(values)
        local_k = min(k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]


    # def get_model_state(self):
    #     saving_dict = {}
    #     saving_dict['_neighbors'] = self._neighbors
    #     saving_dict['_similarity'] = self._similarity
    #     saving_dict['_num_neighbors'] = self._num_neighbors
    #     return saving_dict
    #
    # def set_model_state(self, saving_dict):
    #     self._neighbors = saving_dict['_neighbors']
    #     self._similarity = saving_dict['_similarity']
    #     self._num_neighbors = saving_dict['_num_neighbors']
    #
    # def load_weights(self, path):
    #     with open(path, "rb") as f:
    #         self.set_model_state(pickle.load(f))
    #
    # def save_weights(self, path):
    #     with open(path, "wb") as f:
    #         pickle.dump(self.get_model_state(), f)
