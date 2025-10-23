from functools import cached_property

import numpy as np
import torch
import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow import keras
from torch import nn
from torch_sparse import SparseTensor

from elliot.dataset.samplers.base_sampler import FakeSampler
from elliot.recommender.utils import ModelType


class AbstractRecommender(ABC):
    type: ModelType

    def __init__(self, data, params, seed, logger):
        self._data = data
        self._params = params
        self._users = data.users
        self._items = data.items
        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self.transactions = data.transactions
        self.logger = logger
        self.seed = seed

        self.auto_set_params()
        if hasattr(self, '_loader') or hasattr(self, '_loaders'):
            self.set_side_info()

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def predict(self, start, stop):
        raise NotImplementedError()

    def auto_set_params(self):
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
        params_list = self.params_list if hasattr(self, 'params_list') else []
        for variable_name, public_name, shortcut, default, reading_function, _ in params_list:
            if reading_function is None:
                setattr(self, variable_name, getattr(self._params, public_name, default))
            else:
                setattr(self, variable_name, reading_function(getattr(self._params, public_name, default)))
            self.logger.info(f"Parameter {public_name} set to {getattr(self, variable_name)}")
        if not params_list:
            self.logger.info("No parameters defined")

    def set_side_info(self, loader=None, mod=None):
        name = f"_side{('_' + mod) if mod else ''}"
        loader_name = loader if loader else self._loader
        loader_obj = getattr(self._data.side_information, loader_name)
        setattr(self, name, loader_obj)


class Recommender(AbstractRecommender):
    type = ModelType.BASE

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger)
        np.random.seed(seed)

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError()


class TraditionalRecommender(Recommender):
    type = ModelType.TRADITIONAL

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger)
        self.sampler = FakeSampler()
        self._similarity_matrix = None
        self._preds = None

    def train_step(self, batch):
        pass

    @abstractmethod
    def initialize(self):
        raise NotImplementedError()


class GeneralRecommender(nn.Module, AbstractRecommender):
    type = ModelType.GENERAL

    def __init__(self, data, params, seed, logger):
        AbstractRecommender.__init__(self, data, params, seed, logger)
        super(GeneralRecommender, self).__init__()
        torch.manual_seed(seed)

    @cached_property
    def _sp_i_train(self):
        coo = self._data.sp_i_train.tocoo()
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        return SparseTensor(row=row, col=col, sparse_sizes=coo.shape)

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError()
