import torch
import numpy as np
from scipy.sparse import csr_matrix
import scipy
from scipy.sparse.linalg import svds

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class GFCF(RecMixin, BaseRecommenderModel):
    r"""
    How Powerful is Graph Convolution for Recommendation?

    For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/3459637.3482264>`_

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 512, int, None),
            ("_svd_factors", "svd_factors", "svd_factors", 256, int, None),
            ("_alpha", "alpha", "alpha", 0.3, float, None)
        ]

        self.autoset_params()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.adj_mat = data.sp_i_train.tolil()
        self.d_mat_i, self.d_mat_i_inv, self.vt, self.norm_adj = None, None, None, None

    def train(self):
        adj_mat = self.adj_mat
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = scipy.sparse.diags(d_inv)
        self.norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = scipy.sparse.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = scipy.sparse.diags(1 / d_inv)
        norm_adj = self.norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsc()
        _, _, self.vt = svds(norm_adj, self._svd_factors)

        self.evaluate()

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_eval)):
            offset_stop = min(offset + self._batch_eval, self._num_users)
            predictions = self.get_users_rating(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_users_rating(self, batch_start, batch_stop):
        norm_adj = self.norm_adj
        adj_mat = self.adj_mat
        batch_test = np.array(adj_mat[batch_start: batch_stop, :].todense())
        U_2 = batch_test @ norm_adj.T @ norm_adj
        U_1 = batch_test @ self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
        ret = U_2 + self._alpha * U_1
        return ret

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), torch.tensor(preds).to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    @property
    def name(self):
        return "GFCF" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"
