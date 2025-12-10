"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'


import numpy as np
import torch
from sklearn.utils.extmath import safe_sparse_dot
from tqdm import tqdm

from elliot.recommender.base_recommender import TraditionalRecommender


class EASER(TraditionalRecommender):
    # Model hyperparameters
    l2_norm: float = 1000

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger)

    def initialize(self):
        fake_iter = tqdm(desc="Setting up")

        S = safe_sparse_dot(self._train.T, self._train, dense_output=True)

        diagonal_indices = np.diag_indices(S.shape[0])
        S[diagonal_indices] += self.l2_norm

        fake_iter.set_description("Computing")
        P = np.linalg.inv(S)
        similarity_matrix = P / (-np.diag(P))
        fake_iter.set_description("Done")

        similarity_matrix[diagonal_indices] = 0.0

        self.similarity_matrix = similarity_matrix

    def predict_full(self, user_indices):
        predictions = self._train[user_indices.numpy()] @ self.similarity_matrix

        predictions = torch.from_numpy(predictions)
        return predictions

    def predict_sampled(self, user_indices, item_indices):
        predictions = self._train[user_indices.numpy()] @ self.similarity_matrix

        predictions = torch.from_numpy(predictions)
        predictions = predictions.gather(1, item_indices.clamp(min=0))
        return predictions
