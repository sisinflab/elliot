import torch

from elliot.recommender.base_recommender import TraditionalRecommender
from elliot.recommender.knn.similarity import Similarity


class KNN(TraditionalRecommender):
    # Model hyperparameters
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

        self._backend = Similarity(
            train_data=train_data,
            similarity=self.similarity,
            num_neighbors=self.neighborhood,
            asymmetric_alpha=self.asymmetric_alpha,
            alpha=self.alpha,
            beta=self.beta
        )

        self.neighborhood = self._backend.num_neighbors

        self.params_to_save = ['similarity', 'neighborhood', 'implicit']

    def initialize(self):
        self.similarity_matrix = self._backend.compute_similarity()


class ItemKNN(KNN):
    # Model hyperparameters
    neighborhood: int = 40
    similarity: str = "cosine"
    implicit: bool = False
    asymmetric_alpha: float = 0.5
    alpha: float = 1.0
    beta: float = 1.0

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger, transpose=True)

    def predict_full(self, user_indices):
        predictions = self._URM[user_indices.numpy()] @ self.similarity_matrix

        predictions = torch.from_numpy(predictions.toarray())
        return predictions

    def predict_sampled(self, user_indices, item_indices):
        predictions = self._URM[user_indices.numpy()] @ self.similarity_matrix

        predictions = torch.from_numpy(predictions.toarray())
        predictions = predictions.gather(1, item_indices.clamp(min=0))
        return predictions


class UserKNN(KNN):
    # Model hyperparameters
    neighborhood: int = 40
    similarity: str = "cosine"
    implicit: bool = False
    asymmetric_alpha: float = 0.5
    alpha: float = 1.0
    beta: float = 1.0

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger, transpose=False)

    def predict_full(self, user_indices):
        predictions = self.similarity_matrix[user_indices.numpy()] @ self._URM

        predictions = torch.from_numpy(predictions.toarray())
        return predictions

    def predict_sampled(self, user_indices, item_indices):
        predictions = self.similarity_matrix[user_indices.numpy()] @ self._URM

        predictions = torch.from_numpy(predictions.toarray())
        predictions = predictions.gather(1, item_indices.clamp(min=0))
        return predictions
