import pickle

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, haversine_distances, chi2_kernel, manhattan_distances
from sklearn.metrics import pairwise_distances


class Similarity(object):
    """
    Simple kNN class
    """

    def __init__(self, data, num_neighbors, similarity, implicit):
        self._data = data
        self._ratings = data.train_dict
        self._num_neighbors = num_neighbors
        self._similarity = similarity
        self._implicit = implicit

        if self._implicit:
            self._URM = self._data.sp_i_train
        else:
            self._URM = self._data.sp_i_train_ratings

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

        self.supported_similarities = ["cosine", "dot", ]
        self.supported_dissimilarities = ["euclidean", "manhattan", "haversine",  "chi2", 'cityblock', 'l1', 'l2', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
        print(f"\nSupported Similarities: {self.supported_similarities}")
        print(f"Supported Distances/Dissimilarities: {self.supported_dissimilarities}\n")

        # self._user_ratings = self._ratings
        #
        # self._item_ratings = {}
        # for u, user_items in self._ratings.items():
        #     for i, v in user_items.items():
        #         self._item_ratings.setdefault(i, {}).update({u: v})
        #
        # self._transactions = self._data.transactions

        self._similarity_matrix = np.empty((len(self._users), len(self._users)))

        self.process_similarity(self._similarity)

        ##############
        data, rows_indices, cols_indptr = [], [], []

        column_row_index = np.arange(len(self._users), dtype=np.int32)

        for user_idx in range(len(self._users)):
            cols_indptr.append(len(data))
            column_data = self._similarity_matrix[:, user_idx]

            non_zero_data = column_data != 0

            idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
            top_k_idx = idx_sorted[-self._num_neighbors:]

            data.extend(column_data[non_zero_data][top_k_idx])
            rows_indices.extend(column_row_index[non_zero_data][top_k_idx])

        cols_indptr.append(len(data))

        W_sparse = sparse.csc_matrix((data, rows_indices, cols_indptr),
                                     shape=(len(self._users), len(self._users)), dtype=np.float32).tocsr()
        self._preds = W_sparse.dot(self._URM).toarray()
        ##############
        # self.compute_neighbors()

        del self._similarity_matrix

    # def compute_neighbors(self):
    #     self._neighbors = {}
    #     for x in range(self._similarity_matrix.shape[0]):
    #         arr = np.concatenate((self._similarity_matrix[0:x, x], [-np.inf], self._similarity_matrix[x, x+1:]))
    #         top_indices = np.argpartition(arr, -self._num_neighbors)[-self._num_neighbors:]
    #         arr = arr[top_indices]
    #         self._neighbors[self._private_users[x]] = {self._private_users[i]: arr[p] for p, i in enumerate(top_indices)}

    # def get_user_neighbors(self, item):
    #     return self._neighbors.get(item, {})

    def process_similarity(self, similarity):
        if similarity == "cosine":
            self._similarity_matrix = cosine_similarity(self._URM)
        elif similarity == "dot":
            self._similarity_matrix = (self._URM @ self._URM.T).toarray()
        elif similarity == "euclidean":
            self._similarity_matrix = (1 / (1 + euclidean_distances(self._URM)))
        elif similarity == "manhattan":
            self._similarity_matrix = (1 / (1 + manhattan_distances(self._URM)))
        elif similarity == "haversine":
            self._similarity_matrix = (1 / (1 + haversine_distances(self._URM)))
        elif similarity == "chi2":
            self._similarity_matrix = (1 / (1 + chi2_kernel(self._URM)))
        elif similarity in ['cityblock', 'l1', 'l2']:
            self._similarity_matrix = (1 / (1 + pairwise_distances(self._URM, metric=similarity)))
        elif similarity in ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:
            self._similarity_matrix = (1 / (1 + pairwise_distances(self._URM.toarray(), metric=similarity)))
        else:
            raise ValueError("Compute Similarity: value for parameter 'similarity' not recognized."
                             f"\nAllowed values are: {self.supported_similarities}, {self.supported_dissimilarities}."
                             f"\nPassed value was {similarity}\nTry with implementation: aiolli")

    # def process_cosine(self):
    #     x, y = np.triu_indices(self._similarity_matrix.shape[0], k=1)
    #     self._similarity_matrix[x, y] = cosine_similarity(self._data.sp_i_train_ratings)[x, y]
    #     # g = np.vectorize(self.compute_cosine)
    #     # g(x, y)
    #     # for item_row in range(self._similarity_matrix.shape[0]):
    #     #     for item_col in range(item_row + 1, self._similarity_matrix.shape[1]):
    #     #         self._similarity_matrix[item_row, item_col] = self.compute_cosine(
    #     #             self._item_ratings.get(self._private_items[item_row],{}), self._item_ratings.get(self._private_items[item_col], {}))

    # def compute_cosine(self, i_index, j_index):
    #     u_dict = self._user_ratings.get(self._private_users[i_index],{})
    #     v_dict = self._user_ratings.get(self._private_users[j_index],{})
    #     union_keyset = set().union(*[u_dict, v_dict])
    #     u: np.ndarray = np.array([[1 if x in u_dict.keys() else 0 for x in union_keyset]])
    #     v: np.ndarray = np.array([[1 if x in v_dict.keys() else 0 for x in union_keyset]])
    #     self._similarity_matrix[i_index, j_index] = cosine_similarity(u, v)[0, 0]

    # def get_transactions(self):
    #     return self._transactions

    def get_user_recs(self, u, mask, k):
        user_id = self._data.public_users.get(u)
        user_recs = self._preds[user_id]
        # user_items = self._ratings[u].keys()
        user_recs_mask = mask[user_id]
        user_recs[~user_recs_mask] = -np.inf
        indices, values = zip(*[(self._data.private_items.get(u_list[0]), u_list[1])
                              for u_list in enumerate(user_recs)])

        # indices, values = zip(*predictions.items())
        indices = np.array(indices)
        values = np.array(values)
        local_k = min(k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]

    # def get_user_recs(self, u, mask, k):
    #     user_items = self._ratings[u].keys()
    #     user_mask = mask[self._data.public_users[u]]
    #     predictions = {i: self.score_item(self.get_user_neighbors(u), user_items) for i in self._data.items if
    #                    user_mask[self._data.public_items[i]]}
    #
    #     # user_items = self._ratings[u].keys()
    #     # predictions = {i: self.score_item(self.get_user_neighbors(u), self._item_ratings[i].keys())
    #     #                for i in self._data.items if i not in user_items}
    #
    #     indices, values = zip(*predictions.items())
    #     indices = np.array(indices)
    #     values = np.array(values)
    #     local_k = min(k, len(values))
    #     partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
    #     real_values = values[partially_ordered_preds_indices]
    #     real_indices = indices[partially_ordered_preds_indices]
    #     local_top_k = real_values.argsort()[::-1]
    #     return [(real_indices[item], real_values[item]) for item in local_top_k]

    # @staticmethod
    # def score_item(neighs, user_neighs_items):
    #     num = sum([v for k, v in neighs.items() if k in user_neighs_items])
    #     den = sum(np.power(list(neighs.values()), 1))
    #     return num/den if den != 0 else 0
    def get_model_state(self):
        saving_dict = {}
        saving_dict['_preds'] = self._preds
        saving_dict['_similarity'] = self._similarity
        saving_dict['_num_neighbors'] = self._num_neighbors
        saving_dict['_implicit'] = self._implicit
        return saving_dict

    def set_model_state(self, saving_dict):
        self._preds = saving_dict['_preds']
        self._similarity = saving_dict['_similarity']
        self._num_neighbors = saving_dict['_num_neighbors']
        self._implicit = saving_dict['_implicit']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
