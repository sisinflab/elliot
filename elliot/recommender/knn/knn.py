from abc import abstractmethod

from elliot.recommender.base_recommender import TraditionalRecommender
from elliot.recommender.knn.similarity import Similarity


class KNN(TraditionalRecommender):
    num_neighbors: int
    similarity: str
    implicit: bool
    asymmetric_alpha: float
    alpha: float
    beta: float

    def __init__(self, data, params, seed, logger, transpose):
        super().__init__(data, params, seed, logger)

        self._URM = data.sp_i_train if self.implicit else data.sp_i_train_ratings
        train_data = self._URM if not transpose else self._URM.T

        self._backend = Similarity(train_data=train_data,
                                   similarity=self.similarity,
                                   num_neighbors=self.num_neighbors,
                                   asymmetric_alpha=self.asymmetric_alpha,
                                   alpha=self.alpha,
                                   beta=self.beta)

        self.params_to_save = ['_preds', 'similarity', 'num_neighbors', 'implicit']

    @abstractmethod
    def initialize(self):
        raise NotImplementedError()

    def predict(self, start, stop):
        return self._preds[start:stop]


class ItemKNN(KNN):
    num_neighbors: int = 40
    similarity: str = "cosine"
    implicit: bool = False
    asymmetric_alpha: float = 0.5
    alpha: float = 1.0
    beta: float = 1.0

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger, transpose=True)

    def initialize(self):
        self._similarity_matrix = self._backend.compute_similarity().transpose()
        self._preds = self._URM.dot(self._similarity_matrix)


class UserKNN(KNN):
    num_neighbors: int = 40
    similarity: str = "cosine"
    implicit: bool = False
    asymmetric_alpha: float = 0.5
    alpha: float = 1.0
    beta: float = 1.0

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger, transpose=False)

    def initialize(self):
        self._similarity_matrix = self._backend.compute_similarity()
        self._preds = self._similarity_matrix.dot(self._URM)
