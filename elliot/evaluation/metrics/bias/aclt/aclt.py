"""
This is the implementation of the Average coverage of long tail items metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import operator

import numpy as np
from elliot.evaluation.metrics.base_metric import BaseMetric


class ACLT(BaseMetric):
    """
    This class represents the implementation of the Average coverage of long tail items recommendation metric.
    Passing 'ACLT' to the metrics list will enable the computation of the metric.

    Himan Abdollahpouri, Robin Burke, Bamshad Mobasher
    "Managing Popularity Bias in Recommender Systems with Personalized Re-Ranking"
    Proceedings of the Thirty-Second International Florida Artificial
    Intelligence Research Society Conference, 2019
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
        self._long_tail = self._evaluation_objects.pop.get_long_tail()

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "ACLT"

    @staticmethod
    def __user_aclt(user_recommendations, cutoff, long_tail):
        """
        Per User Average coverage of long tail items
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Average Recommendation Popularity metric for the specific user
        """
        return len(set([i for i,v in user_recommendations[:cutoff]]) & set(long_tail))

    # def eval(self):
    #     """
    #     Evaluation function
    #     :return: the overall averaged value of ACLT
    #     """
    #     return np.average(
    #         [ACLT.__user_aclt(u_r, self._cutoff, self._long_tail)
    #          for u, u_r in self._recommendations.items()]
    #     )

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of ACLT
        """
        return {u: ACLT.__user_aclt(u_r, self._cutoff, self._long_tail)
             for u, u_r in self._recommendations.items()}

