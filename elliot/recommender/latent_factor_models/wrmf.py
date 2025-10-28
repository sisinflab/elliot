"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from tqdm import tqdm

from elliot.dataset.samplers.base_sampler import FakeSampler
from elliot.recommender.base_recommender import Recommender


class WRMF(Recommender):
    """
    Weighted XXX Matrix Factorization

    For further details, please refer to the `paper <https://archive.siam.org/meetings/sdm06/proceedings/059zhangs2.pdf>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        alpha:
        reg: Regularization coefficient

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        WRMF:
          meta:
            save_recs: True
          epochs: 10
          factors: 50
          alpha: 1
          reg: 0.1
    """

    def __init__(self, data, params, seed, logger):
        self.params_list = [
            ("_factors", "factors", "factors", 10, None, None),
            ("_alpha", "alpha", "alpha", 1, None, None),
            ("_reg", "reg", "reg", 0.1, None, None)
        ]
        self.sampler = FakeSampler()
        super().__init__(data, params, seed, logger)

        self.random = np.random
        self.C = self._alpha * self._data.sp_i_train

        self.X = self.random.normal(scale=0.01, size=(self._num_users, self._factors))
        self.Y = self.random.normal(scale=0.01, size=(self._num_items, self._factors))

        self.lambda_eye = self._reg * np.eye(self._factors)

        self.params_to_save = ['X', 'Y', 'C']

    def train_step(self, *args):
        C = self.C
        C_ = C.tocsc()
        YtY = self.Y.T @ self.Y
        XtX = self.X.T @ self.X

        # === USER UPDATES ===
        for u in tqdm(range(self._num_users), desc="Updating users"):
            row_start, row_end = C.indptr[u], C.indptr[u + 1]
            item_idx = C.indices[row_start:row_end]
            Cu = C.data[row_start:row_end]

            if len(item_idx) == 0:
                continue

            Y_u = self.Y[item_idx]

            A = YtY + (Y_u.T * Cu) @ Y_u + self.lambda_eye
            b = (Y_u.T * (Cu + 1.0)) @ np.ones_like(Cu)
            self.X[u] = np.linalg.solve(A, b)

        # === ITEM UPDATES ===
        for i in tqdm(range(self._num_items), desc="Updating items"):
            col_start, col_end = C_.indptr[i], C_.indptr[i + 1]
            user_idx = C_.indices[col_start:col_end]
            Ci = C_.data[col_start:col_end]

            if len(user_idx) == 0:
                continue

            X_i = self.X[user_idx]

            A = XtX + (X_i.T * Ci) @ X_i + self.lambda_eye
            b = (X_i.T * (Ci + 1.0)) @ np.ones_like(Ci)
            self.Y[i] = np.linalg.solve(A, b)

        return 0

    def predict(self, start, stop):
        return self.X[start:stop] @ self.Y.T
