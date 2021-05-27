import pickle

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, haversine_distances, chi2_kernel, \
    manhattan_distances


class Similarity(object):
    """
    Simple VSM class
    """

    def __init__(self, data, user_profile_matrix, item_attribute_matrix, similarity):
        self._data = data
        self._ratings = data.train_dict
        self._user_profile_matrix = user_profile_matrix
        self._item_attribute_matrix = item_attribute_matrix
        self._similarity = similarity

        self._users = self._data.users
        self._items = self._data.items
        self._private_users = self._data.private_users
        self._public_users = self._data.public_users
        self._private_items = self._data.private_items
        self._public_items = self._data.public_items

    def initialize(self):
        """
        This function initialize the data model
        """

        supported_similarities = ["cosine", "dot", ]
        supported_dissimilarities = ["euclidean", "manhattan", "haversine",  "chi2", 'cityblock', 'l1', 'l2', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
        print(f"\nSupported Similarities: {supported_similarities}")
        print(f"Supported Distances/Dissimilarities: {supported_dissimilarities}\n")

        self._transactions = self._data.transactions
        self._similarity_matrix = np.empty((len(self._users), len(self._items)))

        self.process_similarity(self._similarity)

    def process_similarity(self, similarity):
        if similarity == "cosine":
            self._similarity_matrix = cosine_similarity(self._user_profile_matrix, self._item_attribute_matrix)
        elif similarity == "dot":
            self._similarity_matrix = (self._data.sp_i_train_ratings @ self._data.sp_i_train_ratings.T).toarray()
        elif similarity == "euclidean":
            self._similarity_matrix = (1 / (1 + euclidean_distances(self._user_profile_matrix, self._item_attribute_matrix)))
        elif similarity == "manhattan":
            self._similarity_matrix = (1 / (1 + manhattan_distances(self._user_profile_matrix, self._item_attribute_matrix)))
        elif similarity == "haversine":
            self._similarity_matrix = (1 / (1 + haversine_distances(self._user_profile_matrix, self._item_attribute_matrix)))
        elif similarity == "chi2":
            self._similarity_matrix = (1 / (1 + chi2_kernel(self._user_profile_matrix, self._item_attribute_matrix)))
        elif similarity in ['cityblock', 'l1', 'l2']:
            self._similarity_matrix = (1 / (1 + pairwise_distances(self._user_profile_matrix, self._item_attribute_matrix, metric=similarity)))
        elif similarity in ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:
            self._similarity_matrix = (1 / (1 + pairwise_distances(self._user_profile_matrix.toarray(), self._item_attribute_matrix.toarray(), metric=similarity)))
        else:
            raise Exception("Not implemented similarity")

    def get_user_recs(self, u, mask, k):
        user_id = self._data.public_users.get(u)
        user_recs = self._similarity_matrix[user_id]
        user_recs_mask = mask[user_id]
        user_recs[~user_recs_mask] = -np.inf
        indices, values = zip(*[(self._data.private_items.get(u_list[0]), u_list[1])
                              for u_list in enumerate(user_recs)])
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
        saving_dict['_similarity'] = self._similarity
        return saving_dict

    def set_model_state(self, saving_dict):
        self._similarity = saving_dict['_similarity']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)