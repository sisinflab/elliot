"""
This is the implementation of the Precision metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.1'
__author__ = 'XXX'

import numpy as np


class Precision:
    """
    This class represents the implementation of the Precision recommendation metric.
    Passing 'Precision' to the metrics list will enable the computation of the metric.
    """

    def __init__(self, recommendations, config, params):
        """
        Constructor
        :param recommendations: list of recommendations in the form {user: [(item1,value1),...]}
        :param cutoff: numerical threshold to limit the recommendation list
        :param relevant_items: list of relevant items (binary) per user in the form {user: [item1,...]}
        """
        self.recommendations = recommendations
        self.cutoff = config.top_k
        self.relevant_items = params.relevant_items

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "Precision"

    @staticmethod
    def __user_precision(user_recommendations, cutoff, user_relevant_items):
        """
        Per User Precision
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        return sum([1 for i in user_recommendations if i[0] in user_relevant_items]) / cutoff

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of Precision
        """
        return np.average(
            [Precision.__user_precision(u_r, self.cutoff, self.relevant_items[u])
             for u, u_r in self.recommendations.items()]
        )
