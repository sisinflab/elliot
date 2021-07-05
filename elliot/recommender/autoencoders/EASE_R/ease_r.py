"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import time

import numpy as np
from sklearn.utils.extmath import safe_sparse_dot

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class EASER(RecMixin, BaseRecommenderModel):

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_neighborhood", "neighborhood", "neighborhood", -1, int, None),
            ("_l2_norm", "l2_norm", "l2_norm", 1e3, float, None)
        ]

        self.autoset_params()
        if self._neighborhood == -1:
            self._neighborhood = self._data.num_items

    @property
    def name(self):
        return f"EASER_{self.get_params_shortcut()}"

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k):
        return {u: self.get_user_predictions(u, mask, k) for u in self._data.train_dict.keys()}

    def get_user_predictions(self, user_id, mask, top_k=10):
        user_id = self._data.public_users.get(user_id)
        b = self._preds[user_id]
        a = mask[user_id]
        b[~a] = -np.inf
        indices, values = zip(*[(self._data.private_items.get(u_list[0]), u_list[1])
                              for u_list in enumerate(b.data)])

        indices = np.array(indices)
        values = np.array(values)
        local_k = min(top_k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]

    def train(self):
        if self._restore:
            return self.restore_weights()


        start = time.time()

        self._train = self._data.sp_i_train_ratings

        self._similarity_matrix = safe_sparse_dot(self._train.T, self._train, dense_output=True)

        diagonal_indices = np.diag_indices(self._similarity_matrix.shape[0])
        item_popularity = np.ediff1d(self._train.tocsc().indptr)
        self._similarity_matrix[diagonal_indices] = item_popularity + self._l2_norm

        P = np.linalg.inv(self._similarity_matrix)

        self._similarity_matrix = P / (-np.diag(P))

        self._similarity_matrix[diagonal_indices] = 0.0

        end = time.time()
        self.logger.info(f"The similarity computation has taken: {end - start}")

        self._preds = self._train.dot(self._similarity_matrix)

        self.evaluate()
