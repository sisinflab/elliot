from types import SimpleNamespace
import typing as t

from hyperopt import STATUS_OK


class ModelCoordinator(object):
    """
    This class handles the selection of hyperparameters for the hyperparameter tuning realized with HyperOpt.
    """

    def __init__(self, base: SimpleNamespace, params: SimpleNamespace, model_class: t.ClassVar):
        """
        The constructor creates a Placeholder of the recommender model.

        :param base: a SimpleNamespace that contains the configuration (main level) options
        :param params: a SimpleNamespace that contains the hyper-parameters of the model
        :param model_class: the class of the recommendation model
        """
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
        model_params = SimpleNamespace(**self.params.__dict__)

        print("\n************")
        print("Hyperparameter tuning exploration:")
        for k, v in sampled_namespace.__dict__.items():
            model_params.__setattr__(k, v)
            print(f"{k} set to {model_params.__getattribute__(k)}")
        print("************\n")

        model = self.model_class(config=self.base, params=model_params)
        model.train()
        return {
            'loss': -model.get_loss(),
            'status': STATUS_OK,
            'params': model.get_params(),
            'results': model.get_results()
        }
