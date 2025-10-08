import pickle
from abc import abstractmethod

from elliot.recommender.base_recommender import TraditionalRecommender
from elliot.recommender.knn.similarity import Similarity


class KNN(TraditionalRecommender):
    def __init__(self, data, params, seed, logger, transpose):
        self._params_list = [
            ("_num_neighbors", "neighbors", "nn", 40, int, None),
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_implicit", "implicit", "bin", False, None, None),
            ("_asymmetric_alpha", "asymmetric_alpha", "asymalpha", False, None, lambda x: x if x else ""),
            ("_tversky_alpha", "tversky_alpha", "tvalpha", False, None, lambda x: x if x else ""),
            ("_tversky_beta", "tversky_beta", "tvbeta", False, None, lambda x: x if x else "")
        ]
        super().__init__(data, params, seed, logger)

        self._URM = data.sp_i_train if self._implicit else data.sp_i_train_ratings
        train_data = self._URM if not transpose else self._URM.T

        self._backend = Similarity(train_data=train_data,
                                   similarity=self._similarity,
                                   num_neighbors=self._num_neighbors,
                                   asymmetric_alpha=self._asymmetric_alpha,
                                   tversky_alpha=self._tversky_alpha,
                                   tversky_beta=self._tversky_beta)

    @abstractmethod
    def initialize(self):
        raise NotImplementedError()

    def predict(self, start, stop):
        return self._preds[start:stop]

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


class ItemKNN(KNN):
    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger, transpose=True)

    def initialize(self):
        w_sparse = self._backend.compute_similarity()
        self._preds = self._URM.dot(w_sparse)


class UserKNN(KNN):
    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger, transpose=False)

    def initialize(self):
        w_sparse = self._backend.compute_similarity()
        self._preds = w_sparse.dot(self._URM)
