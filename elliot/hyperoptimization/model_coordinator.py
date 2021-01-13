"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from types import SimpleNamespace
import typing as t
import numpy as np

from hyperopt import STATUS_OK


class ModelCoordinator(object):
    """
    This class handles the selection of hyperparameters for the hyperparameter tuning realized with HyperOpt.
    """

    def __init__(self, data_objs, base: SimpleNamespace, params, model_class: t.ClassVar):
        """
        The constructor creates a Placeholder of the recommender model.

        :param base: a SimpleNamespace that contains the configuration (main level) options
        :param params: a SimpleNamespace that contains the hyper-parameters of the model
        :param model_class: the class of the recommendation model
        """
        self.data_objs = data_objs
        self.base = base
        self.params = params
        self.model_class = model_class

    def objective(self, args):
        """
        This function respect the signature, and the return format required for HyperOpt optimization
        :param args: a Dictionary that contains the new hyper-parameter values that will be used in the current run
        :return: it returns a Dictionary with loss, and status being required by HyperOpt,
        and params, and results being required by the framework
        """
        sampled_namespace = SimpleNamespace(**args)
        model_params = SimpleNamespace(**self.params[0].__dict__)

        print("\n************")
        print("Hyperparameter tuning exploration:")
        for k, v in sampled_namespace.__dict__.items():
            model_params.__setattr__(k, v)
            print(f"{k} set to {model_params.__getattribute__(k)}")
        print("************\n")

        losses = []
        results = []
        test_results = []
        for data_obj in self.data_objs:
            model = self.model_class(data=data_obj, config=self.base, params=model_params)
            model.train()
            losses.append(model.get_loss())
            results.append(model.get_results())
            test_results.append(model.get_test_results())

        loss = np.average(losses)
        results = {metric: np.average([result[metric] for result in results]) for metric in model.get_results().keys()}
        test_results = {metric: np.average([result[metric] for result in test_results]) for metric in model.get_results().keys()}

        return {
            'loss': loss,
            'status': STATUS_OK,
            'params': model.get_params(),
            'results': results,
            'statistical_results': model.get_statistical_results(),
            'test_results': test_results,
            'test_statistical_results': model.get_test_statistical_results(),
            'name': model.name
        }

    def single(self):
        """
        This function respect the signature, and the return format required for HyperOpt optimization
        :param args: a Dictionary that contains the new hyper-parameter values that will be used in the current run
        :return: it returns a Dictionary with loss, and status being required by HyperOpt,
        and params, and results being required by the framework
        """

        losses = []
        results = []
        test_results = []
        for data_obj in self.data_objs:
            model = self.model_class(data=data_obj, config=self.base, params=self.params)
            model.train()
            losses.append(model.get_loss())
            results.append(model.get_results())
            test_results.append(model.get_test_results())

        loss = np.average(losses)
        results = {metric: np.average([result[metric] for result in results]) for metric in model.get_results().keys()}
        test_results = {metric: np.average([result[metric] for result in test_results]) for metric in model.get_test_results().keys()}

        return {
            'loss': loss,
            'status': STATUS_OK,
            'params': model.get_params(),
            'results': results,
            'statistical_results': model.get_statistical_results(),
            'test_results': test_results,
            'test_statistical_results': model.get_test_statistical_results(),
            'name': model.name
        }
