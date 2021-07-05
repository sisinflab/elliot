"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pickle
import time
import typing as t
import scipy.sparse as sp

from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.knn.attribute_user_knn.attribute_user_knn_similarity import Similarity
from elliot.recommender.knn.attribute_user_knn.tfidf_utils import TFIDF
from elliot.recommender.base_recommender_model import init_charger


class AttributeUserKNN(RecMixin, BaseRecommenderModel):
    r"""
    Attribute User-kNN proposed in MyMediaLite Recommender System Library

    For further details, please refer to the `paper <https://www.researchgate.net/publication/221141162_MyMediaLite_A_free_recommender_system_library>`_

    Args:
        neighbors: Number of item neighbors
        similarity: Similarity function
        profile: Profile type ('binary', 'tfidf')

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        AttributeUserKNN:
          meta:
            save_recs: True
          neighbors: 40
          similarity: cosine
          profile: binary
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_num_neighbors", "neighbors", "nn", 40, int, None),
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_profile_type", "profile", "profile", "binary", None, None),
            ("_implicit", "implicit", "bin", False, None, None),
            ("_loader", "loader", "load", "ItemAttributes", None, None),
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict

        self._side = getattr(self._data.side_information, self._loader, None)

        if self._profile_type == "tfidf":
            self._tfidf_obj = TFIDF(self._side.feature_map)
            self._tfidf = self._tfidf_obj.tfidf()
            self._user_profiles = self._tfidf_obj.get_profiles(self._ratings)
        else:
            self._user_profiles = {user: self.compute_binary_profile(user_items) for user, user_items in self._ratings.items()}

        self._i_feature_dict = {self._data.public_users[user]: {self._side.public_features[feature]: value
                                                                for feature, value in user_features.items()}
                                for user, user_features in self._user_profiles.items()}
        self._sp_i_features = self.build_feature_sparse_values()

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

    @property
    def name(self):
        return f"AttributeUserKNN_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        start = time.time()
        self._model.initialize()
        end = time.time()
        print(f"The similarity computation has taken: {end - start}")

        print(f"Transactions: {self._data.transactions}")


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


    def compute_binary_profile(self, user_items_dict: t.Dict):
        user_features = {}
        partial = 1/len(user_items_dict)
        for item in user_items_dict.keys():
            for feature in self._side.feature_map.get(item,[]):
                user_features[feature] = user_features.get(feature, 0) + partial
        return user_features

    def build_feature_sparse(self):

        rows_cols = [(i, f) for i, features in self._i_feature_dict.items() for f in features]
        rows = [u for u, _ in rows_cols]
        cols = [i for _, i in rows_cols]
        data = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                             shape=(self._num_items, len(self._side.public_features)))
        return data

    def build_feature_sparse_values(self):
        rows_cols_values = [(u, f, v) for u, features in self._i_feature_dict.items() for f, v in features.items()]
        rows = [u for u, _, _ in rows_cols_values]
        cols = [i for _, i, _ in rows_cols_values]
        values = [r for _, _, r in rows_cols_values]

        data = sp.csr_matrix((values, (rows, cols)), dtype='float32',
                             shape=(self._num_users, len(self._side.public_features)))

        return data
