import ast
import pickle
import random

import numpy as np
import torch

from torch import nn, Tensor
from torch_sparse import SparseTensor
from abc import ABC, abstractmethod
from functools import cached_property

from elliot.dataset.samplers.base_sampler import FakeSampler
from elliot.recommender.init import zeros_init
from elliot.recommender.utils import ModelType, device


class AbstractRecommender(ABC):
    type: ModelType

    def __init__(self, data, params, seed, logger):
        self._data = data
        self._seed = seed
        self._users = data.users
        self._items = data.items
        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self.transactions = data.transactions
        self.logger = logger
        self.params_list = []
        self.params_to_save = []

        self.set_seed(seed)
        self.init_params(params)

        if hasattr(self, '_loader') or hasattr(self, '_loaders'):
            self.set_side_info()
        if hasattr(self, 'sampler'):
            self.sampler.events = data.transactions

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
    def predict(self, start, stop):
        raise NotImplementedError()

    @abstractmethod
    def set_seed(self, seed: int):
        raise NotImplementedError()

    def init_params(self, params):
        """
        Define Parameters as tuples: (variable_name, public_name, shortcut, default, reading_function, printing_function)
        Example:

        self._params_list = [
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_user_profile_type", "user_profile", "up", "tfidf", None, None),
            ("_item_profile_type", "item_profile", "ip", "tfidf", None, None),
            ("_mlpunits", "mlp_units", "mlpunits", "(1,2,3)", lambda x: list(make_tuple(x)), lambda x: str(x).replace(",", "-")),
        ]
        """
        self.logger.info("Loading parameters")

        for ann, data_type in self.__class__.__annotations__.items():
            value = getattr(params, ann, getattr(self, ann))
            if data_type == tuple and isinstance(value, str):
                value = ast.literal_eval(value)
            setattr(self, ann, data_type(value))
            self.logger.info(f"Parameter {ann} set to {value}")
            self.params_list.append(ann)

        self.params_to_save = self.params_list.copy()
        # params_list = self.params_list if hasattr(self, 'params_list') else []
        # for variable_name, public_name, shortcut, default, reading_function, _ in params_list:
        #     if reading_function is None:
        #         setattr(self, variable_name, getattr(self._params, public_name, default))
        #     else:
        #         setattr(self, variable_name, reading_function(getattr(self._params, public_name, default)))
        #     self.logger.info(f"Parameter {public_name} set to {getattr(self, variable_name)}")
        # if not params_list:
        #     self.logger.info("No parameters defined")
        # self.params_to_save = params_list

    def set_side_info(self, loader=None, mod=None):
        name = f"_side{('_' + mod) if mod else ''}"
        loader_name = loader if loader else self._loader
        loader_obj = getattr(self._data.side_information, loader_name)
        setattr(self, name, loader_obj)

    @abstractmethod
    def save_weights(self, path):
        raise NotImplementedError()

    @abstractmethod
    def load_weights(self, path):
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

    @abstractmethod
    def train_step(self, *args):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, start, stop):
        raise NotImplementedError()

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self._get_model_state(), f)

    def load_weights(self, path):
        with open(path, "rb") as f:
            self._set_model_state(pickle.load(f))

    def _get_model_state(self):
        return {p[0]: getattr(self, p[0]) for p in self.params_to_save}

    def _set_model_state(self, saving_dict):
        for k, v in saving_dict:
            if k in self.params_to_save:
                setattr(self, k, v)


class TraditionalRecommender(Recommender):
    type = ModelType.TRADITIONAL

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger)
        self.sampler = FakeSampler()
        self.similarity_matrix = None
        self._train = data.sp_i_train_ratings
        self._implicit_train = data.sp_i_train

    def train_step(self, *args):
        pass

    @abstractmethod
    def initialize(self):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, start, stop):
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

    @cached_property
    def _sp_i_train(self):
        coo = self._data.sp_i_train.tocoo()
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        return SparseTensor(row=row, col=col, sparse_sizes=coo.shape)

    @abstractmethod
    def train_step(self, batch, *args):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, start, stop):
        raise NotImplementedError()

    def save_weights(self, file_path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, file_path)

    def load_weights(self, file_path):
        checkpoint = torch.load(file_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


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

    @abstractmethod
    def train_step(self, batch, *args):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, start, stop):
        raise NotImplementedError()
