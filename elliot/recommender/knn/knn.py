import torch
from sklearn.preprocessing import normalize

from elliot.dataset import Interactions
from elliot.namespace import RecommenderConfig
from elliot.recommender.base_recommender import TraditionalRecommender
from elliot.recommender.knn.similarity import Similarity
from elliot.utils.registry import model_registry


class KNN(TraditionalRecommender):
    # Model hyperparameters
    neighborhood: int
    similarity: str
    implicit: bool
    asymmetric_alpha: float
    alpha: float
    beta: float
    normalize_similarity: bool

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        transpose: bool,
        *args,
        **kwargs
    ):
        super().__init__(params, seed, interactions, *args, **kwargs)

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

        if self.normalize_similarity:
            self.similarity_matrix = normalize(self.similarity_matrix, norm="l1", axis=1)


@model_registry.register()
class ItemKNN(KNN):
    # Model hyperparameters
    neighborhood: int = 40
    similarity: str = "cosine"
    implicit: bool = False
    asymmetric_alpha: float = 0.5
    alpha: float = 1.0
    beta: float = 1.0
    normalize_similarity: bool = False

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        *args,
        **kwargs
    ):
        super().__init__(params, seed, interactions, *args, **kwargs, transpose=True)

    def predict(self, user_indices, item_indices=None, **kwargs):
        predictions = self._URM[user_indices.numpy()] @ self.similarity_matrix

        predictions = torch.from_numpy(predictions.toarray())

        if item_indices is None:
            return predictions

        predictions = predictions.gather(1, item_indices.clamp(min=0))
        return predictions


@model_registry.register()
class UserKNN(KNN):
    # Model hyperparameters
    neighborhood: int = 40
    similarity: str = "cosine"
    implicit: bool = False
    asymmetric_alpha: float = 0.5
    alpha: float = 1.0
    beta: float = 1.0
    normalize_similarity: bool = False

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        *args,
        **kwargs
    ):
        super().__init__(params, seed, interactions, *args, **kwargs, transpose=False)

    def predict(self, user_indices, item_indices=None, **kwargs):
        predictions = self.similarity_matrix[user_indices.numpy()] @ self._URM

        predictions = torch.from_numpy(predictions.toarray())

        if item_indices is None:
            return predictions

        predictions = predictions.gather(1, item_indices.clamp(min=0))
        return predictions
