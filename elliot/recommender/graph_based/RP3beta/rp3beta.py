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

    def __init__(self, data, params, seed, logger):
        self.params_list = [
            ("_neighborhood", "neighborhood", "neighborhood", 10, int, None),
            ("_alpha", "alpha", "alpha", 1., float, None),
            ("_beta", "beta", "beta", 0.6, float, None),
            ("_normalize_similarity", "normalize_similarity", "normalize_similarity", False, bool, None)
        ]
        super().__init__(data, params, seed, logger)

        self._train = data.sp_i_train_ratings
        self._implicit_train = data.sp_i_train

        if self._neighborhood == -1:
            self._neighborhood = self._data.num_items

    def predict(self, start, stop):
        return self._preds[start:stop]

    def initialize(self):
        w_sparse = self._compute_similarity()
        self._preds = self._train.dot(w_sparse)

    def _compute_similarity(self):
        # Neighborhood along rows
        Pui = normalize(self._train, norm='l1', axis=1)

        X_bool = self._implicit_train.transpose(copy=True)
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()
        Piu = normalize(X_bool, norm='l1', axis=1)
        del X_bool

        Pui = Pui.power(self._alpha)
        Piu = Piu.power(self._alpha)

        degree = np.where(X_bool_sum != 0.0, X_bool_sum ** (-self._beta), 0.0)
        D = sp.diags(degree)

        block_dim = 200
        rows, cols, values = [], [], []

        with tqdm(total=(Pui.shape[1] // block_dim), desc="Computing") as t:
            for start in range(0, Pui.shape[1], block_dim):
                end = min(start + block_dim, Pui.shape[1])
                d_t_block = Piu[start:end, :]

                similarity_block = d_t_block.dot(Pui).dot(D).toarray()
                np.fill_diagonal(similarity_block[:, start:end], 0.0)
                idx = np.argpartition(similarity_block, -self._neighborhood, axis=1)[:, -self._neighborhood:]
                top_vals = np.take_along_axis(similarity_block, idx, axis=1)

                for i in range(similarity_block.shape[0]):
                    mask = top_vals[i] != 0.0
                    cols_r = idx[i][mask]
                    vals_r = top_vals[i][mask]
                    if len(vals_r) > 0:
                        rows.extend([start + i] * len(vals_r))
                        cols.extend(cols_r)
                        values.extend(vals_r)

                t.update()

            rows = np.array(rows)
            cols = np.array(cols)
            values = np.array(values)

            t.set_description("Done")
            t.refresh()

        similarity_matrix = sparse.create_sparse_matrix(
            data=(values, (rows, cols)), dtype=np.float32, shape=(Pui.shape[1], Pui.shape[1])
        )

        if self._normalize_similarity:
            similarity_matrix = normalize(similarity_matrix, norm='l1', axis=1)

        return similarity_matrix

    def _process_along_cols(self, similarity_matrix):
        # Neighborhood along cols
        matrix = similarity_matrix.tocsc()

        data, rows_indices, cols_indptr = [], [], [0]

        for item_idx in range(matrix.shape[1]):
            start_position = matrix.indptr[item_idx]
            end_position = matrix.indptr[item_idx + 1]

            column_data = matrix.data[start_position:end_position]
            column_row_index = matrix.indices[start_position:end_position]

            non_zero_data = column_data != 0

            idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
            top_k_idx = idx_sorted[-self._neighborhood:]

            data.extend(column_data[non_zero_data][top_k_idx])
            rows_indices.extend(column_row_index[non_zero_data][top_k_idx])
            cols_indptr.append(len(data))

        return sparse.create_sparse_matrix(
            data=(data, rows_indices, cols_indptr), shape=matrix.shape, dtype=np.float32
        ).tocsr()
