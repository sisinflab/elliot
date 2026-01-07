"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from scipy.sparse import csr_matrix, diags
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize

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
        degree = np.zeros_like(X_bool_sum, dtype=float)
        mask = X_bool_sum != 0.0
        degree[mask] = np.power(X_bool_sum[mask], -self.beta)
        D = diags(degree)

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

    def _compute_blockwise_similarity(self, Piu, Pui, D):
        rows, cols, values = [], [], []
        block_dim = Pui.shape[1] // 50

        t = tqdm(total=(Pui.shape[1] // block_dim))
        t.set_description("Computing")

        # Process matrix in blocks along rows
        for start in range(0, Pui.shape[1], block_dim):
            end = min(start + block_dim, Pui.shape[1])
            d_t_block = Piu[start:end, :]

            # Compute similarity block matrix product
            similarity_block = (d_t_block @ Pui @ D)#.toarray()

            # Set to 0 self-similarity entries (diagonal elements)
            b_rows, b_cols = similarity_block.nonzero()
            b_mask = b_rows == b_cols
            similarity_block.data[b_mask] = 0.0

            # Apply sparse top-k row filtering
            b_rows, b_cols, b_data = self._get_top_k(similarity_block, start)

            # Store computed values
            rows.append(b_rows)
            cols.append(b_cols)
            values.append(b_data)

            t.update()

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        values = np.concatenate(values)

        t.set_description("Build csr matrix")

        # Create final sparse matrix from accumulated values
        sim_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(Pui.shape[1], Pui.shape[1]),
            dtype=np.float32
        )

        t.set_description("Done")
        t.refresh()

        return sim_matrix

    def _get_top_k(self, block, start):
        rows_list = []
        cols_list = []
        vals_list = []

        for i in range(block.shape[0]):
            # Get row data and indices
            row_data = block.data[block.indptr[i]:block.indptr[i + 1]]
            row_idx = block.indices[block.indptr[i]:block.indptr[i + 1]]

            # Skip rows with only 0 elements
            if len(row_data) == 0:
                continue

            # Determine actual number of elements to keep
            k = min(self.neighborhood, len(row_data))

            # Find indices of top-k largest values using argpartition
            topk_idx = np.argpartition(row_data, -k)[-k:]

            # Store (row, col) indices and row values
            rows_list.extend([i + start] * len(topk_idx))
            cols_list.extend(row_idx[topk_idx])
            vals_list.extend(row_data[topk_idx])

        return (
            np.array(rows_list),
            np.array(cols_list),
            np.array(vals_list)
        )

    def predict_full(self, user_indices):
        predictions = self._train[user_indices.numpy()] @ self.similarity_matrix

        predictions = torch.from_numpy(predictions.toarray())
        return predictions

    def predict_sampled(self, user_indices, item_indices):
        predictions = self._train[user_indices.numpy()] @ self.similarity_matrix

        predictions = torch.from_numpy(predictions.toarray())
        predictions = predictions.gather(1, item_indices.clamp(min=0))
        return predictions
