"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pickle
import time
import scipy.sparse as sparse
from sklearn.metrics.pairwise import cosine_similarity
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation
from sklearn.preprocessing import normalize
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.NN.item_knn.item_knn_similarity import Similarity
from elliot.recommender.NN.item_knn.aiolli_ferrari import AiolliSimilarity
from elliot.recommender.base_recommender_model import init_charger

np.random.seed(42)


class RP3beta(RecMixin, BaseRecommenderModel):

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._random = np.random

        self._params_list = [
            ("_neighborhood", "neighborhood", "neighborhood", 10, int, None),
            ("_alpha", "alpha", "alpha", 1., float, None),
            ("_beta", "beta", "beta", 0.6, float, None),
            ("_min_rating", "min_rating", "min_rating", 0, float, None),
            ("_implicit", "implicit", "implicit", False, bool, None),
            ("_normalize_similarity", "normalize_similarity", "normalize_similarity", False, bool, None)
        ]

        self.autoset_params()

        self._train = self._data.sp_i_train_ratings

        if self._min_rating > 0:
            self._train.data[self._min_rating >= self._train.data] = 0
            self._train.eliminate_zeros()
            if self._implicit:
                self._train.data = np.ones(self._train.data.size, dtype=np.float32)

        self.Pui = normalize(self._data.sp_i_train_ratings, norm='l1', axis=1)

        X_bool = self._train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()

        self.degree = np.zeros(self._train.shape[1])

        nonZeroMask = X_bool_sum != 0.0

        self.degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self._beta)

        self.Piu = normalize(X_bool, norm='l1', axis=1)
        del (X_bool)

        if self._alpha != 1.:
            self.Pui = self.Pui.power(self._alpha)
            self.Piu = self.Piu.power(self._alpha)

    @property
    def name(self):
        return f"RP3beta_{self.get_params_shortcut()}"

    def get_recommendations(self, k: int = 10):
        return {u: self.get_user_predictions(u, self._similarity_matrix) for u in self._data.train_dict.keys()}

    def get_user_predictions(self, user_id, W_sparse):
        user_id = self._data.public_users.get(user_id)
        b = self._data.sp_i_train_ratings[user_id].dot(W_sparse)
        a = self.get_train_mask(user_id, user_id+1)
        b[a] = -np.inf
        indices, values = zip(*[(self._data.private_items.get(u_list[0]), u_list[1])
                              for u_list in enumerate(b.data)])

        indices = np.array(indices)
        values = np.array(values)
        local_k = min(self.evaluator.get_needed_recommendations(), len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]

    def train(self):
        if self._restore:
            return self.restore_weights()

        block_dim = 200
        d_t = self.Piu

        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start = time.time()

        for current_block_start_row in range(0, self.Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > self.Pui.shape[1]:
                block_dim = self.Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * self.Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = np.multiply(similarity_block[row_in_block, :], self.degree)
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self._neighborhood]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

            # if time.time() - start_time_printBatch > 60:
            #     self._print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
            #         current_block_start_row,
            #         100.0 * float(current_block_start_row) / self.Pui.shape[1],
            #         (time.time() - start_time) / 60,
            #         float(current_block_start_row) / (time.time() - start_time)))
            #
            #
            #     start_time_printBatch = time.time()

        self._similarity_matrix = sparse.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(self.Pui.shape[1], self.Pui.shape[1]))

        if self._normalize_similarity:
            self._similarity_matrix = normalize(self._similarity_matrix, norm='l1', axis=1)

        end = time.time()
        print(f"The similarity computation has taken: {end - start}")

        best_metric_value = 0

        recs = self.get_recommendations(self._neighborhood)
        result_dict = self.evaluator.eval(recs)
        self._results.append(result_dict)
        print(f'Finished')

        if self._results[-1][self._validation_k]["val_results"][self._validation_metric] > best_metric_value:
            print("******************************************")
            if self._save_weights:
                with open(self._saving_filepath, "wb") as f:
                    pickle.dump(self._model.get_model_state(), f)
            if self._save_recs:
                store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")