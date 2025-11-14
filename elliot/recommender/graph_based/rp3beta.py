"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.preprocessing import normalize

from elliot.utils import sparse
from elliot.recommender.base_recommender import TraditionalRecommender


class RP3beta(TraditionalRecommender):
    # Model hyperparameters
    neighborhood: int = 10
    alpha: float = 1.0
    beta: float = 0.6
    normalize_similarity: bool = False

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger)

        if self.neighborhood == -1:
            self.neighborhood = self._data.num_items

    def initialize(self):
        # Step 1: Normalize user-item matrix
        Pui = normalize(self._train, norm="l1", axis=1)

        # Step 2: Get boolean item-user matrix
        X_bool = self._implicit_train.transpose(copy=True)

        # Step 3: Calculate item popularity degrees
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()
        degree = np.where(X_bool_sum != 0.0, np.power(X_bool_sum, -self.beta), 0.0)
        D = sp.diags(degree)

        # Step 4: Normalize item-user matrix
        Piu = normalize(X_bool, norm="l1", axis=1)
        del X_bool

        # Apply alpha exponent
        if self.alpha != 1.0:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Step 5: Compute similarity in blocks and apply top-k filtering
        similarity_matrix = self._compute_blockwise_similarity(Piu, Pui, D)

        # Step 6: Normalize if required
        if self.normalize_similarity:
            similarity_matrix = normalize(similarity_matrix, norm="l1", axis=1)

        # Store computed similarity matrix
        self.similarity_matrix = similarity_matrix

    def _compute_blockwise_similarity(self, Piu, Pui, D, block_dim=200):
        rows, cols, values = [], [], []

        with tqdm(total=(Pui.shape[1] // block_dim), desc="Computing") as t:
            # Process matrix in blocks along rows
            for start in range(0, Pui.shape[1], block_dim):
                end = min(start + block_dim, Pui.shape[1])
                d_t_block = Piu[start:end, :]

                # Compute similarity block matrix product
                similarity_block = (d_t_block @ Pui @ D).toarray()

                # Set to 0 self-similarity entries (diagonal elements)
                np.fill_diagonal(similarity_block[:, start:end], 0.0)

                # Apply sparse top-k row filtering
                b_rows, b_cols, b_data = self._get_top_k(similarity_block, start)

                # Store computed values
                rows.extend(b_rows)
                cols.extend(b_cols)
                values.extend(b_data)

                t.update()

            rows = np.array(rows)
            cols = np.array(cols)
            values = np.array(values)

            t.set_description("Done")
            t.refresh()

        # Create final sparse matrix from accumulated values
        return sparse.create_sparse_matrix(
            data=(values, (rows, cols)), dtype=np.float32, shape=(Pui.shape[1], Pui.shape[1])
        )

    def _get_top_k(self, block, start):
        b_row, b_col, b_data = [], [], []

        # Process each row individually
        for i in range(block.shape[0]):
            row = block[i]
            cols = np.nonzero(row)[0]

            if not len(cols):
                continue

            # Determine actual number of elements to keep
            top_k = min(self.neighborhood, len(cols))

            # Find indices of top-k largest values using argpartition
            idx = np.argpartition(row, -top_k)[-top_k:]

            # Store top-k values in their original column positions
            values = row[idx]

            b_row.extend([i + start] * len(idx))
            b_col.extend(idx)
            b_data.extend(values)

        return b_row, b_col, b_data

    def predict(self, start, stop):
        predictions = self._train[start:stop] @ self.similarity_matrix
        return predictions
