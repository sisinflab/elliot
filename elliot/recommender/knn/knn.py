from abc import abstractmethod

from elliot.recommender.base_recommender import TraditionalRecommender
from elliot.recommender.knn.similarity import Similarity


class KNN(TraditionalRecommender):
    neighborhood: int
    similarity: str
    implicit: bool
    asymmetric_alpha: float
    alpha: float
    beta: float

    def __init__(self, data, params, seed, logger, transpose):
        super().__init__(data, params, seed, logger)

        self._URM = self._implicit_train if self.implicit else self._train
        train_data = self._URM if not transpose else self._URM.T

        self._backend = Similarity(train_data=train_data,
                                   similarity=self.similarity,
                                   num_neighbors=self.neighborhood,
                                   asymmetric_alpha=self.asymmetric_alpha,
                                   alpha=self.alpha,
                                   beta=self.beta)

        self.params_to_save = ['similarity', 'num_neighbors', 'implicit']

    def initialize(self):
        self.similarity_matrix = self._backend.compute_similarity()

    @abstractmethod
    def predict(self, start, stop):
        raise NotImplementedError()


class ItemKNN(KNN):
    neighborhood: int = 40
    similarity: str = "cosine"
    implicit: bool = False
    asymmetric_alpha: float = 0.5
    alpha: float = 1.0
    beta: float = 1.0

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger, transpose=True)

    def predict(self, start, stop):
        predictions = self._URM[start:stop] @ self.similarity_matrix.T
        return predictions


class UserKNN(KNN):
    neighborhood: int = 40
    similarity: str = "cosine"
    implicit: bool = False
    asymmetric_alpha: float = 0.5
    alpha: float = 1.0
    beta: float = 1.0

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger, transpose=False)

    def predict(self, start, stop):
        predictions = self.similarity_matrix[start:stop] @ self._URM
        return predictions
