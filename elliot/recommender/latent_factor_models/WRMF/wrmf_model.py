"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve


class WRMFModel(object):
    """
    Simple Matrix Factorization class
    """

    def __init__(self, factors, data, random, alpha, reg):

        self._data = data
        self.random = random
        self.C = alpha * self._data.sp_i_train
        self.train_dict = self._data.train_dict
        self.user_num, self.item_num = self._data.num_users, self._data.num_items

        self.X = sp.csr_matrix(self.random.normal(scale=0.01,
                                                  size=(self.user_num, factors)))
        self.Y = sp.csr_matrix(self.random.normal(scale=0.01,
                                                  size=(self.item_num, factors)))
        self.X_eye = sp.eye(self.user_num)
        self.Y_eye = sp.eye(self.item_num)
        self.lambda_eye = reg * sp.eye(factors)

        self.user_vec, self.item_vec, self.pred_mat = None, None, None

    def train_step(self):
        yTy = self.Y.T.dot(self.Y)
        xTx = self.X.T.dot(self.X)
        for u in range(self.user_num):
            Cu = self.C[u, :].toarray()
            Pu = Cu.copy()
            Pu[Pu != 0] = 1
            CuI = sp.diags(Cu, [0])
            yTCuIY = self.Y.T.dot(CuI).dot(self.Y)
            yTCuPu = self.Y.T.dot(CuI + self.Y_eye).dot(Pu.T)
            self.X[u] = spsolve(yTy + yTCuIY + self.lambda_eye, yTCuPu)
        for i in range(self.item_num):
            Ci = self.C[:, i].T.toarray()
            Pi = Ci.copy()
            Pi[Pi != 0] = 1
            CiI = sp.diags(Ci, [0])
            xTCiIX = self.X.T.dot(CiI).dot(self.X)
            xTCiPi = self.X.T.dot(CiI + self.X_eye).dot(Pi.T)
            self.Y[i] = spsolve(xTx + xTCiIX + self.lambda_eye, xTCiPi)

        self.pred_mat = self.X.dot(self.Y.T).A

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

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
