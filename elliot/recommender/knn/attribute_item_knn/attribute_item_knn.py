"""
Module description:

"""


import numpy as np
import pickle
import time

from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation
import scipy.sparse as sp

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.knn.attribute_item_knn.attribute_item_knn_similarity import Similarity
from elliot.recommender.base_recommender_model import init_charger


class AttributeItemKNN(RecMixin, BaseRecommenderModel):
    r"""
    Attribute Item-kNN proposed in MyMediaLite Recommender System Library

    For further details, please refer to the `paper <https://www.researchgate.net/publication/221141162_MyMediaLite_A_free_recommender_system_library>`_

    Args:
        neighbors: Number of item neighbors
        similarity: Similarity function

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        AttributeItemKNN:
          meta:
            save_recs: True
          neighbors: 40
          similarity: cosine
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_num_neighbors", "neighbors", "nn", 40, int, None),
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_implicit", "implicit", "bin", False, None, None),
            ("_loader", "loader", "load", "ItemAttributes", None, None),
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict

        self._side = getattr(self._data.side_information, self._loader, None)

        self._i_feature_dict = {i_item: [self._side.public_features[feature] for feature
                                         in self._side.feature_map[item]] for item, i_item
                                in self._data.public_items.items()}
        self._sp_i_features = self.build_feature_sparse()

        self._sp_i_features = self._sp_i_features * 1
        self._sp_i_features = sp.hstack([self._sp_i_features, self._data.sp_i_train_ratings.T], format='csr')

        self._sp_i_features = self.TF_IDF(self._sp_i_features)

        self._model = Similarity(data=self._data, attribute_matrix=self._sp_i_features, num_neighbors=self._num_neighbors, similarity=self._similarity, implicit=self._implicit)

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_recs(u, mask, k) for u in self._ratings.keys()}

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def build_feature_sparse(self):

        rows_cols = [(i, f) for i, features in self._i_feature_dict.items() for f in features]
        rows = [u for u, _ in rows_cols]
        cols = [i for _, i in rows_cols]
        data = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                             shape=(self._num_items, len(self._side.public_features)))
        return data

    @property
    def name(self):
        return f"AttributeItemKNN_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        start = time.time()
        self._model.initialize()
        end = time.time()
        self.logger.info(
            "Similarity computation completed",
            extra={"context": {"duration_sec": end - start}}
        )

        self.logger.info(
            "Dataset summary",
            extra={"context": {"transactions": self._data.transactions}}
        )

        self.evaluate()

        # best_metric_value = 0
        #
        # recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
        # result_dict = self.evaluator.eval(recs)
        # self._results.append(result_dict)
        # print(f'Finished')
        #
        # if self._results[-1][self._validation_k]["val_results"][self._validation_metric] > best_metric_value:
        #     print("******************************************")
        #     if self._save_weights:
        #         with open(self._saving_filepath, "wb") as f:
        #             pickle.dump(self._model.get_model_state(), f)
        #     if self._save_recs:
        #         store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")

    def okapi_BM_25(self, dataMatrix, K1=1.2, B=0.75):
        """
        Items are assumed to be on rows
        :param dataMatrix:
        :param K1:
        :param B:
        :return:
        """

        assert B > 0 and B < 1, "okapi_BM_25: B must be in (0,1)"
        assert K1 > 0, "okapi_BM_25: K1 must be > 0"

        assert np.all(np.isfinite(dataMatrix.data)), \
            "okapi_BM_25: Data matrix contains {} non finite values".format(
                np.sum(np.logical_not(np.isfinite(dataMatrix.data))))

        # Weighs each row of a sparse matrix by OkapiBM25 weighting
        # calculate idf per term (user)

        dataMatrix = sp.coo_matrix(dataMatrix)

        N = float(dataMatrix.shape[0])
        idf = np.log(N / (1 + np.bincount(dataMatrix.col)))

        # calculate length_norm per document
        row_sums = np.ravel(dataMatrix.sum(axis=1))

        average_length = row_sums.mean()
        length_norm = (1.0 - B) + B * row_sums / average_length

        # weight matrix rows by bm25
        denominator = K1 * length_norm[dataMatrix.row] + dataMatrix.data
        denominator[denominator == 0.0] += 1e-9

        dataMatrix.data = dataMatrix.data * (K1 + 1.0) / denominator * idf[dataMatrix.col]

        return dataMatrix.tocsr()

    def TF_IDF(self, dataMatrix):
        """
        Items are assumed to be on rows
        :param dataMatrix:
        :return:
        """

        assert np.all(np.isfinite(dataMatrix.data)), \
            "TF_IDF: Data matrix contains {} non finite values.".format(
                np.sum(np.logical_not(np.isfinite(dataMatrix.data))))

        assert np.all(dataMatrix.data >= 0.0), \
            "TF_IDF: Data matrix contains {} negative values, computing the square root is not possible.".format(
                np.sum(dataMatrix.data < 0.0))

        # TFIDF each row of a sparse amtrix
        dataMatrix = sp.coo_matrix(dataMatrix)
        N = float(dataMatrix.shape[0])

        # calculate IDF
        idf = np.log(N / (1 + np.bincount(dataMatrix.col)))

        # apply TF-IDF adjustment
        dataMatrix.data = np.sqrt(dataMatrix.data) * idf[dataMatrix.col]

        return dataMatrix.tocsr()