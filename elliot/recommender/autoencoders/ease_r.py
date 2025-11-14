"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'


import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from tqdm import tqdm

from elliot.recommender.base_recommender import TraditionalRecommender


class EASER(TraditionalRecommender):
    neighborhood: int = -1
    l2_norm: float = 1e3

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger)

        if self.neighborhood == -1:
            self.neighborhood = self._data.num_items

    def initialize(self):
        fake_iter = tqdm(range(1), desc="Computing")
        similarity_matrix, diagonal_indices = None, None

        for _ in fake_iter:
            S = safe_sparse_dot(self._train.T, self._train, dense_output=True)

            diagonal_indices = np.diag_indices(S.shape[0])
            item_popularity = np.ediff1d(self._train.tocsc().indptr)
            S[diagonal_indices] = item_popularity + self.l2_norm

            P = np.linalg.inv(S)
            similarity_matrix = P / (-np.diag(P))
            fake_iter.set_description("Done")

        similarity_matrix[diagonal_indices] = 0.0

        self.similarity_matrix = similarity_matrix

    def predict(self, start, stop):
        predictions = self._train[start:stop] @ self.similarity_matrix
        return predictions
