from abc import ABC
from abc import abstractmethod
import numpy as np
import random


class BaseRecommenderModel(ABC):
    def __init__(self, config, params, *args, **kwargs):
        """
        This class represents a recommender model. You can load a pretrained model
        by specifying its checkpoint path and use it for training/testing purposes.

        Args:
            data: data loader object
            params: dictionary with all parameters
        """
        self._config = config
        self._params = params

        self._restore_epochs = getattr(self._params, "restore_epoch", -1)
        self._validation_metric = getattr(self._params, "validation_metric", "nDCG")
        self._save_weights = getattr(self._params, "save_weights", False)
        self._save_recs = getattr(self._params, "save_recs", False)
        self._verbose = getattr(self._params, "verbose", 1)
        self._results = []

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def get_recommendations(self, *args):
        pass

    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def get_results(self):
        pass
