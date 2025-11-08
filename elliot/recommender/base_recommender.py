import pickle
import random

import numpy as np
import torch

from torch import nn
from torch_sparse import SparseTensor
from abc import ABC, abstractmethod
from functools import cached_property

from elliot.dataset.samplers.base_sampler import FakeSampler
from elliot.recommender.utils import ModelType, xavier_uniform_initialization, xavier_normal_initialization, \
    zeros_initialization, device


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
        self.random = np.random

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

    @abstractmethod
    def train_step(self, *args):
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
        self._similarity_matrix = None
        self._preds = None

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

    @cached_property
    def _sp_i_train(self):
        coo = self._data.sp_i_train.tocoo()
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        return SparseTensor(row=row, col=col, sparse_sizes=coo.shape)

    @abstractmethod
    def train_step(self, batch, *args):
        raise NotImplementedError()

    def _init_weights(self, init_type, modules=None):
        init_types = {
            'xavier_uniform': xavier_uniform_initialization,
            'xavier_normal': xavier_normal_initialization,
            'zeros': zeros_initialization
        }
        modules = modules if modules is not None else self.modules()
        init_func = init_types.get(init_type)
        if init_func is None:
            raise NotImplementedError()
        for m in modules:
            init_func(m)

    def save_weights(self, file_path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, file_path)

    def load_weights(self, file_path):
        checkpoint = torch.load(file_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
