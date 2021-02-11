"""
This is the implementation of the Mean Absolute Error metric.
It proceeds from a system-wise computation.
"""
import warnings

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from elliot.evaluation.metrics.base_metric import BaseMetric


class MAE(BaseMetric):
    """
    This class represents the implementation of the Mean Absolute Error recommendation metric.
    Passing 'MAE' to the metrics list will enable the computation of the metric.
    """

    def __init__(self, recommendations, config, params, eval_objects):
        """
        Constructor
        :param recommendations: list of recommendations in the form {user: [(item1,value1),...]}
        :param config: SimpleNameSpace that represents the configuration of the experiment
        :param params: Parameters of the model
        :param eval_objects: list of objects that may be useful for the computation of the different metrics
        """
        super().__init__(recommendations, config, params, eval_objects)
        self._cutoff = self._evaluation_objects.cutoff
        self._relevant_items = self._evaluation_objects.relevance.get_binary_relevance()
        self._total_relevant_items = sum([len(self._relevant_items[u]) for u, _ in self._recommendations.items()])
        self._test = self._evaluation_objects.relevance.get_test()

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "MAE"

    @staticmethod
    def __user_MAE(user_recommendations, user_test, user_relevant_items):
        """
        Per User computation for Mean Absolute Error
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        return sum([abs(v - user_test[i]) for i, v in user_recommendations if i in user_relevant_items])

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of Mean Absolute Error
        """
        return sum(
            [MAE.__user_MAE(u_r, self._test[u], self._relevant_items[u])
             for u, u_r in self._recommendations.items() if len(self._relevant_items[u])]
        ) / self._total_relevant_items

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of Mean Absolute Error per user
        """
        return {u: MAE.__user_MAE(u_r, self._test[u], self._relevant_items[u])/len(self._relevant_items[u])
             for u, u_r in self._recommendations.items() if len(self._relevant_items[u])}

    @staticmethod
    def needs_full_recommendations():
        warnings.warn("\n*** WARNING: Mean Absolute Error metric requires full length recommendations")
        return True
