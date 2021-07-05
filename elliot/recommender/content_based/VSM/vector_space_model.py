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
from elliot.recommender.content_based.VSM.vector_space_model_similarity import Similarity
from elliot.recommender.content_based.VSM.tfidf_utils import TFIDF
from elliot.recommender.base_recommender_model import init_charger


class VSM(RecMixin, BaseRecommenderModel):
    r"""
    Vector Space Model

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/2362499.2362501>`_ and the `paper <https://ieeexplore.ieee.org/document/9143460>`_

    Args:
        similarity: Similarity metric
        user_profile:
        item_profile:

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        VSM:
          meta:
            save_recs: True
          similarity: cosine
          user_profile: binary
          item_profile: binary
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_user_profile_type", "user_profile", "up", "tfidf", None, None),
            ("_item_profile_type", "item_profile", "ip", "tfidf", None, None),
            ("_loader", "loader", "load", "ItemAttributes", None, None),
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict

        self._side = getattr(self._data.side_information, self._loader, None)

        if self._user_profile_type == "tfidf":
            self._tfidf_obj = TFIDF(self._side.feature_map)
            self._tfidf = self._tfidf_obj.tfidf()
            self._user_profiles = self._tfidf_obj.get_profiles(self._ratings)
        else:
            self._user_profiles = {user: self.compute_binary_profile(user_items)
                                   for user, user_items in self._ratings.items()}

        self._i_user_feature_dict = {self._data.public_users[user]: {self._side.public_features[feature]: value
                                                                     for feature, value in user_features.items()}
                                     for user, user_features in self._user_profiles.items()}
        self._sp_i_user_features = self.build_feature_sparse_values(self._i_user_feature_dict, self._num_users)

        if self._item_profile_type == "tfidf":
            self._tfidf_obj = TFIDF(self._side.feature_map)
            self._tfidf = self._tfidf_obj.tfidf()
            self._i_item_feature_dict = {
                i_item: {self._side.public_features[feature]: self._tfidf[item].get(feature, 0)
                         for feature in self._side.feature_map[item]}
                for item, i_item in self._data.public_items.items()}
            self._sp_i_item_features = self.build_feature_sparse_values(self._i_item_feature_dict, self._num_items)
        else:
            self._i_item_feature_dict = {i_item: [self._side.public_features[feature] for feature
                                                  in self._side.feature_map[item]]
                                         for item, i_item in self._data.public_items.items()}
            self._sp_i_item_features = self.build_feature_sparse(self._i_item_feature_dict, self._num_items)

        self._model = Similarity(self._data, self._sp_i_user_features, self._sp_i_item_features, self._similarity)

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
        return f"VSM_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        start = time.time()
        self._model.initialize()
        end = time.time()
        self.logger.info(f"The similarity computation has taken: {end - start}")

        self.evaluate()

    def compute_binary_profile(self, user_items_dict: t.Dict):
        user_features = {}
        # partial = 1/len(user_items_dict)
        for item in user_items_dict.keys():
            for feature in self._side.feature_map.get(item, []):
                # user_features[feature] = user_features.get(feature, 0) + partial
                user_features[feature] = user_features.get(feature, 1)
        return user_features

    def build_feature_sparse(self, feature_dict, num_entities):

        rows_cols = [(i, f) for i, features in feature_dict.items() for f in features]
        rows = [u for u, _ in rows_cols]
        cols = [i for _, i in rows_cols]
        data = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                             shape=(num_entities, len(self._side.public_features)))
        return data

    def build_feature_sparse_values(self, feature_dict, num_entities):
        rows_cols_values = [(u, f, v) for u, features in feature_dict.items() for f, v in features.items()]
        rows = [u for u, _, _ in rows_cols_values]
        cols = [i for _, i, _ in rows_cols_values]
        values = [r for _, _, r in rows_cols_values]

        data = sp.csr_matrix((values, (rows, cols)), dtype='float32',
                             shape=(num_entities, len(self._side.public_features)))

        return data

