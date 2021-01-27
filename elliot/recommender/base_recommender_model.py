"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from abc import ABC
from abc import abstractmethod


class BaseRecommenderModel(ABC):
    def __init__(self, data, config, params, *args, **kwargs):
        """
        This class represents a recommender model. You can load a pretrained model
        by specifying its checkpoint path and use it for training/testing purposes.

        Args:
            data: data loader object
            params: dictionary with all parameters
        """
        self._data = data
        self._config = config
        self._params = params

        self._restore_epochs = getattr(self._params.meta, "restore_epoch", -1)
        self._validation_metric = getattr(self._params.meta, "validation_metric", "nDCG@10")
        self._validation_metric = self._validation_metric.split("@")
        self._validation_k = int(self._validation_metric[1])
        self._validation_metric = self._validation_metric[0]
        self._save_weights = getattr(self._params.meta, "save_weights", False)
        self._save_recs = getattr(self._params.meta, "save_recs", False)
        self._verbose = getattr(self._params.meta, "verbose", None)
        self._validation_rate = getattr(self._params.meta, "validation_rate", 1)
        self._compute_auc = getattr(self._params.meta, "compute_auc", False)
        self._epochs = getattr(self._params, "epochs", 2)
        if self._epochs < self._validation_rate:
            raise Exception(f"The first validation epoch ({self._validation_rate}) is later than the overall number of epochs ({self._epochs}).")
        self._batch_size = getattr(self._params, "batch_size", -1)
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

    # @abstractmethod
    # def get_statistical_results(self):
    #     pass
    #
    # @abstractmethod
    # def get_test_results(self):
    #     pass
    #
    # @abstractmethod
    # def get_test_statistical_results(self):
    #     pass
