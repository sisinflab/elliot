"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle

import numpy as np
from scipy import sparse as sp


class iALSModel(object):
    """
    Simple Matrix Factorization class
    """

    def __init__(self, factors, data, random, alpha, epsilon, reg, scaling):

        self._data = data
        self.random = random
        self.C = self._data.sp_i_train
        if scaling == "linear":
            self.C.data = 1.0 + alpha * self.C.data
        elif scaling == "log":
            self.C.data = 1.0 + alpha * np.log(1.0 + self.C.data / epsilon)
        self.C_csc = self.C.tocsc()
        self.train_dict = self._data.train_dict
        self.user_num, self.item_num = self._data.num_users, self._data.num_items

        self.X = self.random.normal(scale=0.01, size=(self.user_num, factors))
        self.Y = self.random.normal(scale=0.01, size=(self.item_num, factors))

        warm_item_mask = np.ediff1d(self._data.sp_i_train.tocsc().indptr) > 0
        self.warm_items = np.arange(0, self.item_num, dtype=np.int32)[warm_item_mask]

        self.X_eye = sp.eye(self.user_num)
        self.Y_eye = sp.eye(self.item_num)
        self.lambda_eye = reg * sp.eye(factors)

        self.user_vec, self.item_vec, self.pred_mat = None, None, None

    def train_step(self):
        yTy = self.Y.T.dot(self.Y)

        C = self.C
        for u in range(self.user_num):
            start = C.indptr[u]
            end = C.indptr[u+1]

            Cu = C.data[start:end]
            Pu = self.Y[C.indices[start:end], :]

            B = yTy + Pu.T.dot(((Cu - 1) * Pu.T).T) + self.lambda_eye

            self.X[u] = np.dot(np.linalg.inv(B), Pu.T.dot(Cu))

        xTx = self.X.T.dot(self.X)
        C = self.C_csc
        for i in self.warm_items:
            start = C.indptr[i]
            end = C.indptr[i + 1]

            Cu = C.data[start:end]
            Pi = self.X[C.indices[start:end], :]

            B = xTx + Pi.T.dot(((Cu - 1) * Pi.T).T) + self.lambda_eye

            self.Y[i] = np.dot(np.linalg.inv(B), Pi.T.dot(Cu))

    def predict(self, user, item):
        return self.pred_mat[self._data.public_users[user], self._data.public_items[item]]

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
        saving_dict['pred_mat'] = self.pred_mat
        saving_dict['X'] = self.X
        saving_dict['Y'] = self.Y
        saving_dict['C'] = self.C
        return saving_dict

    def set_model_state(self, saving_dict):
        self.pred_mat = saving_dict['pred_mat']
        self.X = saving_dict['X']
        self.Y = saving_dict['Y']
        self.C = saving_dict['C']

    def prepare_predictions(self):
        self.pred_mat = self.X.dot(self.Y.T)

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
