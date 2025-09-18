import numpy as np

from abc import ABC, abstractmethod
from enum import Enum

from elliot.recommender.base_trainer import GeneralTrainer, TraditionalTrainer


class ModelType(Enum):
    GENERAL = 1
    TRADITIONAL = 2


class AbstractRecommender(ABC):
    type: ModelType

    def __init__(self, data, params, seed, logger, *args):
        self._params_list = []
        self._users = data.users
        self._items = data.items
        self.transactions = data.transactions
        self.logger = logger
        np.random.seed(seed)

    @abstractmethod
    def predict(self, start, stop):
        raise NotImplementedError()

    def auto_set_params(self, params):
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
        for variable_name, public_name, shortcut, default, reading_function, _ in self._params_list:
            if reading_function is None:
                setattr(self, variable_name, getattr(params, public_name, default))
            else:
                setattr(self, variable_name, reading_function(getattr(params, public_name, default)))
            self.logger.info(f"Parameter {public_name} set to {getattr(self, variable_name)}")
        if not self._params_list:
            self.logger.info("No parameters defined")


class GeneralRecommender(AbstractRecommender):
    type = ModelType.GENERAL

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError()


class TraditionalRecommender(AbstractRecommender):
    type = ModelType.TRADITIONAL

    @abstractmethod
    def initialize(self):
        raise NotImplementedError()


def get_model(data, config, params, model_class: AbstractRecommender, *args, **kwargs):
    #model = model_class(data, params)
    if model_class.type == ModelType.GENERAL:
        trainer = GeneralTrainer
    elif model_class.type == ModelType.TRADITIONAL:
        trainer = TraditionalTrainer
    return trainer(data, config, params, model_class, *args, **kwargs)
