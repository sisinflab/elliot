"""
Module description:

"""
import logging as pylog
from elliot.evaluation.evaluator import Evaluator
from elliot.utils.folder import build_model_folder

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from abc import ABC
from abc import abstractmethod
from functools import wraps
from elliot.utils import logging


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

        self._restore = getattr(self._params.meta, "restore", False)

        _cutoff_k = getattr(data.config.evaluation, "cutoffs", [data.config.top_k])
        _cutoff_k = _cutoff_k if isinstance(_cutoff_k, list) else [_cutoff_k]
        _first_metric = data.config.evaluation.simple_metrics[0] if data.config.evaluation.simple_metrics else ""
        _default_validation_k = _cutoff_k[0]
        self._validation_metric = getattr(self._params.meta, "validation_metric",
                                          _first_metric + "@" + str(_default_validation_k)).split("@")
        if self._validation_metric[0].lower() not in [m.lower()
                                                      for m in data.config.evaluation.simple_metrics]:
            raise Exception("Validation metric must be in the list of simple metrics")

        self._validation_k = int(self._validation_metric[1]) if len(self._validation_metric) > 1 else _cutoff_k[0]
        if self._validation_k not in _cutoff_k:
            raise Exception("Validation cutoff must be in general cutoff values")

        self._validation_metric = self._validation_metric[0]
        self._save_weights = getattr(self._params.meta, "save_weights", False)
        self._save_recs = getattr(self._params.meta, "save_recs", False)
        self._verbose = getattr(self._params.meta, "verbose", None)
        self._validation_rate = getattr(self._params.meta, "validation_rate", 1)
        self._compute_auc = getattr(self._params.meta, "compute_auc", False)
        self._epochs = getattr(self._params, "epochs", 2)
        self._iteration = 0
        if self._epochs < self._validation_rate:
            raise Exception(f"The first validation epoch ({self._validation_rate}) "
                            f"is later than the overall number of epochs ({self._epochs}).")
        self._batch_size = getattr(self._params, "batch_size", -1)
        self._results = []
        self._params_list = []

    def get_params_shortcut(self):
        return "_".join([str(p[2])+":"+ str(p[5](getattr(self, p[0])) if p[5] else getattr(self, p[0])) for p in self._params_list])

    def autoset_params(self):
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
                setattr(self, variable_name, getattr(self._params, public_name, default))
            else:
                setattr(self, variable_name, reading_function(getattr(self._params, public_name, default)))
            self.logger.info(f"Parameter {public_name} set to {getattr(self, variable_name)}")
        if not self._params_list:
            self.logger.info("No parameters defined")

    @staticmethod
    def _batch_remove(original_str: str, char_list):
        for c in char_list:
            original_str = original_str.replace(c, "")
        return original_str

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


def init_charger(init):
    @wraps(init)
    def new_init(self, *args, **kwargs):
        BaseRecommenderModel.__init__(self, *args, **kwargs)
        self.logger = logging.get_logger(self.__class__.__name__, pylog.CRITICAL if self._config.config_test else
                                         pylog.DEBUG)
        self._num_items = self._data.num_items
        self._num_users = self._data.num_users

        init(self, *args, **kwargs)

        self.evaluator = Evaluator(self._data, self._params)
        self._params.name = self.name
        build_model_folder(self._config.path_output_rec_weight, self.name)
        self._saving_filepath = f'{self._config.path_output_rec_weight}{self.name}/best-weights-{self.name}'

    return new_init
