import pickle
import random
from types import SimpleNamespace

import numpy as np
import torch

from torch import nn, Tensor
from torch_sparse import SparseTensor
from abc import ABC, abstractmethod

from elliot.recommender.init import zeros_init
from elliot.recommender.utils import ModelType, device
from elliot.utils.config import build_recommender_config


class AbstractRecommender(ABC):
    type: ModelType

    def __init__(self, data, params, seed, logger):
        self._data = data
        self._seed = seed
        self._users, self._items = data.get_users_items()
        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self.transactions = data.transactions
        self.logger = logger
        self.params_list = []
        self.params_to_save = []

        self.set_seed(seed)
        self.set_params(params)

        if hasattr(self, '_loader') or hasattr(self, '_loaders'):
            self.set_side_info()

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def name_param(self):
        """The name of the model with all it's parameters."""
        name = ""
        for ann, _ in self.__class__.__annotations__.items():
            value = getattr(self, ann, None)
            if isinstance(value, float):
                name += f"_{ann}={value:.4f}"
            else:
                name += f"_{ann}={value}"
        return name

    @abstractmethod
    def set_seed(self, seed: int):
        raise NotImplementedError()

    def set_params(self, params: SimpleNamespace):
        self.logger.info("Loading parameters")

        RecommenderConfig = build_recommender_config(self.__class__)
        validator = RecommenderConfig(**vars(params))

        for name, val in validator.get_validated_params().items():
            setattr(self, name, val)
            self.logger.info(f"Parameter {name} set to {val}")
            self.params_list.append(name)

        self.params_to_save = self.params_list.copy()

    def set_side_info(self, loader=None, mod=None):
        name = f"_side{('_' + mod) if mod else ''}"
        loader_name = loader if loader else self._loader
        loader_obj = getattr(self._data.side_information, loader_name)
        setattr(self, name, loader_obj)

    def get_training_dataloader(self, batch_size):
        for _ in range(1):
            yield None

    @abstractmethod
    def train_step(self, batch, *args):
        raise NotImplementedError()

    @abstractmethod
    def predict_full(self, user_indices):
        raise NotImplementedError()

    @abstractmethod
    def predict_sampled(self, user_indices, item_indices):
        raise NotImplementedError()

    @abstractmethod
    def get_model_state(self):
        raise NotImplementedError()

    @abstractmethod
    def set_model_state(self, checkpoint):
        raise NotImplementedError()


class Recommender(AbstractRecommender):
    type = ModelType.BASE

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger)
        self.modules = []
        self.bias = []

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def apply(self, init_func, **kwargs):
        for m in self.modules:
            if any(m is x for x in self.bias):
                zeros_init(m)
            else:
                init_func(m, **kwargs)

    def get_model_state(self):
        return {p[0]: getattr(self, p[0]) for p in self.params_to_save}

    def set_model_state(self, checkpoint):
        for k, v in checkpoint:
            if k in self.params_to_save:
                setattr(self, k, v)


class TraditionalRecommender(Recommender):
    type = ModelType.TRADITIONAL

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger)
        self.similarity_matrix = None
        self._train = data.sp_i_train_ratings
        self._implicit_train = data.sp_i_train

    def train_step(self, *args):
        pass

    @abstractmethod
    def initialize(self):
        raise NotImplementedError()


class GeneralRecommender(nn.Module, AbstractRecommender):
    type = ModelType.GENERAL

    def __init__(self, data, params, seed, logger):
        AbstractRecommender.__init__(self, data, params, seed, logger)
        super(GeneralRecommender, self).__init__()
        self.bias = []
        self._device = device

    def set_seed(self, seed: int):
        """Set random seed for reproducibility.

        Args:
            seed (int): The seed value to be used.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def apply(self, init_func, **kwargs):
        for m in self.modules():
            if any(m is x for x in self.bias):
                zeros_init(m)
            else:
                init_func(m, **kwargs)

    def get_model_state(self):
        return {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }

    def set_model_state(self, checkpoint):
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class GraphBasedRecommender(GeneralRecommender):
    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger)

    def get_adj_mat(self) -> SparseTensor:
        """Get the normalized interaction matrix of users and items.

        Returns:
            SparseTensor: The sparse adjacency matrix.
        """
        # Extract user and items nodes
        row, col = self._data.sp_i_train.nonzero()
        user_nodes = row
        item_nodes = col + self._num_users

        # Unify arcs in both directions
        row = np.concatenate([user_nodes, item_nodes])
        col = np.concatenate([item_nodes, user_nodes])

        # Create the edge tensor
        edge_index_np = np.vstack([row, col])
        # Creating a tensor directly from a numpy array instead of lists
        edge_index = torch.tensor(edge_index_np, dtype=torch.int64)

        size = self._num_items + self._num_users

        # Create the SparseTensor using the edge indexes.
        # This is the format expected by LGConv
        adj = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            sparse_sizes=(size, size),
        ).to(self._device)

        return adj

    def get_ego_embeddings(
        self, user_embedding: nn.Embedding, item_embedding: nn.Embedding
    ) -> Tensor:
        """Get the initial embedding of users and items and combine to an embedding matrix.

        Args:
            user_embedding (nn.Embedding): The user embeddings.
            item_embedding (nn.Embedding): The item embeddings.

        Returns:
            Tensor: Combined user and item embeddings.
        """
        user_embeddings = user_embedding.weight
        item_embeddings = item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings
