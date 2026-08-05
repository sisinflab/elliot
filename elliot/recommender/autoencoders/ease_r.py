"""
Module description:

"""



import numpy as np
import torch
from sklearn.utils.extmath import safe_sparse_dot
from tqdm import tqdm

from elliot.dataset import Interactions
from elliot.namespace import RecommenderConfig
from elliot.recommender.base_recommender import TraditionalRecommender
from elliot.utils.registry import model_registry


@model_registry.register()
class EASER(TraditionalRecommender):
    # Model hyperparameters
    l2_norm: float = 1000

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        *args,
        **kwargs
    ):
        super().__init__(params, seed, interactions, *args, **kwargs)

    def initialize(self):
        t = tqdm()
        t.set_description("Setting up")

        S = safe_sparse_dot(self._train.T, self._train, dense_output=True)

        diagonal_indices = np.diag_indices(S.shape[0])
        S[diagonal_indices] += self.l2_norm

        t.set_description("Computing")

        P = np.linalg.inv(S)
        similarity_matrix = P / (-np.diag(P))

        t.set_description("Done")

        similarity_matrix[diagonal_indices] = 0.0

        self.similarity_matrix = similarity_matrix

    def predict(self, user_indices, item_indices=None, **kwargs):
        predictions = self._train[user_indices.numpy()] @ self.similarity_matrix

        predictions = torch.from_numpy(predictions)

        if item_indices is None:
            return predictions

        predictions = predictions.gather(1, item_indices.clamp(min=0))
        return predictions
