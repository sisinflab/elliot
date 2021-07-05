"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from types import SimpleNamespace
import typing as t
import numpy as np
import logging as pylog

from elliot.utils import logging

from hyperopt import STATUS_OK


class ModelCoordinator(object):
    """
    This class handles the selection of hyperparameters for the hyperparameter tuning realized with HyperOpt.
    """

    def __init__(self, data_objs, base: SimpleNamespace, params, model_class: t.ClassVar, test_fold_index: int):
        """
        The constructor creates a Placeholder of the recommender model.

        :param base: a SimpleNamespace that contains the configuration (main level) options
        :param params: a SimpleNamespace that contains the hyper-parameters of the model
        :param model_class: the class of the recommendation model
        """
        self.logger = logging.get_logger(self.__class__.__name__, pylog.CRITICAL if base.config_test else pylog.DEBUG)
        self.data_objs = data_objs
        self.base = base
        self.params = params
        self.model_class = model_class
        self.test_fold_index = test_fold_index
        self.model_config_index = 0

    def objective(self, args):
        """
        This function respect the signature, and the return format required for HyperOpt optimization
        :param args: a Dictionary that contains the new hyper-parameter values that will be used in the current run
        :return: it returns a Dictionary with loss, and status being required by HyperOpt,
        and params, and results being required by the framework
        """
        sampled_namespace = SimpleNamespace(**args)
        model_params = SimpleNamespace(**self.params[0].__dict__)

        self.logger.info("Hyperparameter tuning exploration:")
        for (k, v) in sampled_namespace.__dict__.items():
            model_params.__setattr__(k, v)
            self.logger.info(f"{k} set to {model_params.__getattribute__(k)}")

        losses = []
        results = []
        for trainval_index, data_obj in enumerate(self.data_objs):
            self.logger.info(f"Exploration: Hyperparameter exploration number {self.model_config_index+1}")
            self.logger.info(f"Exploration: Test Fold exploration number {self.test_fold_index+1}")
            self.logger.info(f"Exploration: Train-Validation Fold exploration number {trainval_index+1}")
            model = self.model_class(data=data_obj, config=self.base, params=model_params)
            model.train()
            losses.append(model.get_loss())
            results.append(model.get_results())

        self.model_config_index += 1

        loss = np.average(losses)
        results = self._average_results(results)

        return {
            'loss': loss,
            'status': STATUS_OK,
            'params': model.get_params(),
            'val_results': {k: result_dict["val_results"] for k, result_dict in results.items()},
            'val_statistical_results': {k: result_dict["val_statistical_results"] for k, result_dict in model.get_results().items()},
            'test_results': {k: result_dict["test_results"] for k, result_dict in results.items()},
            'test_statistical_results': {k: result_dict["test_statistical_results"] for k, result_dict in model.get_results().items()},
            'name': model.name
        }

    def single(self):
        """
        This function respect the signature, and the return format required for HyperOpt optimization
        :param args: a Dictionary that contains the new hyper-parameter values that will be used in the current run
        :return: it returns a Dictionary with loss, and status being required by HyperOpt,
        and params, and results being required by the framework
        """

        self.logger.info("Hyperparameters:")
        for k, v in self.params.__dict__.items():
            self.logger.info(f"{k} set to {v}")

        losses = []
        results = []
        for trainval_index, data_obj in enumerate(self.data_objs):
            self.logger.info(f"Exploration: Test Fold exploration number {self.test_fold_index+1}")
            self.logger.info(f"Exploration: Train-Validation Fold exploration number {trainval_index+1}")
            model = self.model_class(data=data_obj, config=self.base, params=self.params)
            model.train()
            losses.append(model.get_loss())
            results.append(model.get_results())

        loss = np.average(losses)
        results = self._average_results(results)

        return {
            'loss': loss,
            'status': STATUS_OK,
            'params': model.get_params(),
            'val_results': {k: result_dict["val_results"] for k, result_dict in results.items()},
            'val_statistical_results': {k: result_dict["val_statistical_results"] for k, result_dict in model.get_results().items()},
            'test_results': {k: result_dict["test_results"] for k, result_dict in results.items()},
            'test_statistical_results': {k: result_dict["test_statistical_results"] for k, result_dict in model.get_results().items()},
            'name': model.name
        }

    @staticmethod
    def _average_results(results_list):
        ks = list(results_list[0].keys())
        eval_result_types = ["val_results", "test_results"]
        metrics = list(results_list[0][ks[0]]["val_results"].keys())
        return {k: {type_: {metric: np.average([fold_result[k][type_][metric]
                                                for fold_result in results_list])
                            for metric in metrics}
                    for type_ in eval_result_types}
                for k in ks}
