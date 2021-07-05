"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle
import time
import numpy as np
import sys
import scipy.sparse as sp
from sklearn.linear_model import ElasticNet


class SlimModel(object):
    def __init__(self,
                 data, num_users, num_items, l1_ratio, alpha, epochs, neighborhood, random_seed):

        self._data = data
        self._num_users = num_users
        self._num_items = num_items
        self._l1_ratio = l1_ratio
        self._alpha = alpha
        self._epochs = epochs
        self._neighborhood = neighborhood

        self.md = ElasticNet(alpha=self._alpha,
                             l1_ratio=self._l1_ratio,
                             positive=True,
                             fit_intercept=False,
                             copy_X=False,
                             precompute=True,
                             selection='random',
                             max_iter=100,
                             random_state=random_seed,
                             tol=1e-4)

        self._w_sparse = None
        self.pred_mat = None

    def train(self, verbose):
        train = self._data.sp_i_train_ratings

        dataBlock = 10000000

        rows = np.empty(dataBlock, dtype=np.int32)
        cols = np.empty(dataBlock, dtype=np.int32)
        values = np.empty(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        for currentItem in range(self._num_items):
            y = train[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = train.indptr[currentItem]
            end_pos = train.indptr[currentItem + 1]

            current_item_data_backup = train.data[start_pos: end_pos].copy()
            train.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.md.fit(train, y)

            nonzero_model_coef_index = self.md.sparse_coef_.indices
            nonzero_model_coef_value = self.md.sparse_coef_.data

            local_topK = min(len(nonzero_model_coef_value) - 1, self._neighborhood)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):
                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1

            train.data[start_pos:end_pos] = current_item_data_backup

            if verbose and (time.time() - start_time_printBatch > 300 or (
                    currentItem + 1) % 1000 == 0 or currentItem == self._num_items - 1):
                print('{}: Processed {} ( {:.2f}% ) in {:.2f} minutes. Items per second: {:.0f}'.format(
                    'SLIMElasticNetRecommender',
                    currentItem + 1,
                    100.0 * float(currentItem + 1) / self._num_items,
                    (time.time() - start_time) / 60,
                    float(currentItem) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        # generate the sparse weight matrix
        self._w_sparse = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                      shape=(self._num_items, self._num_items), dtype=np.float32)

    def prepare_predictions(self):
        self.pred_mat = self._data.sp_i_train_ratings.dot(self._w_sparse).toarray()

    def predict(self, u, i):
        return self.pred_mat[u, i]

    def get_user_recs(self, user, mask, k=100):
        # user_items = self._data.train_dict[user].keys()
        # predictions = {i: self.predict(user, i) for i in self._data.items if i not in user_items}

        ui = self._data.public_users[user]
        user_mask = mask[ui]
        predictions = {self._data.private_items[i]: self.predict(ui, i) for i in range(self._data.num_items) if user_mask[i]}

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
        saving_dict['_A_tilde'] = self._A_tilde
        return saving_dict

    def set_model_state(self, saving_dict):
        self._A_tilde = saving_dict['_A_tilde']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
