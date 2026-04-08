"""
Module description:

"""


import numpy as np
import torch
from scipy import sparse as sp
from tqdm import tqdm

from elliot.recommender.base_recommender import BaseRecommender
from elliot.recommender.init import normal_init
from elliot.utils.registry import model_registry


@model_registry.register()
class iALS(BaseRecommender):
    """
    Simple Matrix Factorization class
    """

    # Model hyperparameters
    factors: int = 10
    alpha: float = 1.0
    epsilon: float = 1.0
    lambda_weights: float = 0.1
    scaling: str = "linear"

    def __init__(self, params, interactions, seed, *args, **kwargs):
        super().__init__(params, interactions, seed, *args, **kwargs)

        self.C = self._interactions.sparse

        if self.scaling == "linear":
            self.C.data = 1.0 + self.alpha * self.C.data
        elif self.scaling == "log":
            self.C.data = 1.0 + self.alpha * np.log(1.0 + self.C.data / self.epsilon)
        else:
            raise ValueError(f"Unknown option '{self.scaling}' for scaling")

        self.C_csc = self.C.tocsc()

        # Embeddings
        self.X = np.empty((self._num_users, self.factors), dtype=np.float32)
        self.Y = np.empty((self._num_items, self.factors), dtype=np.float32)

        warm_item_mask = np.ediff1d(self.C_csc.indptr) > 0
        self.warm_items = np.arange(0, self._num_items, dtype=np.int32)[warm_item_mask]

        self.X_eye = sp.eye(self._num_users)
        self.Y_eye = sp.eye(self._num_items)
        self.lambda_eye = self.lambda_weights * sp.eye(self.factors)

        # Init embedding weights
        self.modules = [self.X, self.Y]
        self.apply(normal_init)

        self.params_to_save = ['X', 'Y', 'C']

    def train_step(self, *args):
        yTy = self.Y.T.dot(self.Y)

        C = self.C
        for u in tqdm(range(self._num_users), desc="Computing"):
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

        return 0

    def predict(self, user_indices, item_indices=None):
        predictions = self.X[user_indices.numpy()] @ self.Y.T

        predictions = torch.from_numpy(predictions)

        if item_indices is None:
            return predictions

        predictions = predictions.gather(1, item_indices.clamp(min=0))
        return predictions
