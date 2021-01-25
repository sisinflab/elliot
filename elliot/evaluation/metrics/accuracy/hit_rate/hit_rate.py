"""
This is the implementation of the Hit Rate metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import typing as t
from evaluation.metrics.base_metric import BaseMetric


class HR(BaseMetric):
    """
    This class represents the implementation of the Hit Rate recommendation metric.
    Passing 'HR' to the metrics list will enable the computation of the metric.
    """

    def __init__(self, recommendations: t.Dict[int, t.List[t.Tuple[int, float]]], config, params, eval_objects):
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

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "HR"

    @staticmethod
    def __user_HR(user_recommendations, cutoff, user_relevant_items):
        """
        Per User Hit Rate
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        return 1 if sum([1 for i in user_recommendations[:cutoff] if i[0] in user_relevant_items]) > 0 else 0

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of Hit Rate
        """
        return np.average(
            [HR.__user_HR(u_r, self._cutoff, self._relevant_items[u])
             for u, u_r in self._recommendations.items() if len(self._relevant_items[u])]
        )

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of Hit Rate per user
        """
        return {u: HR.__user_HR(u_r, self._cutoff, self._relevant_items[u])
             for u, u_r in self._recommendations.items() if len(self._relevant_items[u])}

