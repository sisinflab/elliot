"""
This is the implementation of the Recall metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.1'
__author__ = 'XXX'

import numpy as np


class Recall:
    """
    This class represents the implementation of the Recall recommendation metric.
    Passing 'Recall' to the metrics list will enable the computation of the metric.
    """

    def __init__(self, recommendations, params):
        """
        Constructor
        :param recommendations: list of recommendations in the form {user: [(item1,value1),...]}
        :param cutoff: numerical threshold to limit the recommendation list
        :param relevant_items: list of relevant items (binary) per user in the form {user: [item1,...]}
        """
        self.recommendations = recommendations
        self.cutoff = params.k
        self.relevant_items = params.relevant_items

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "Recall"

    @staticmethod
    def __user_recall(user_recommendations, cutoff, user_relevant_items):
        """
        Per User Recall
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Recall metric for the specific user
        """
        # TODO check formula
        return sum([1 for i in user_recommendations if i[0] in user_relevant_items]) / \
               min(len(user_relevant_items), cutoff)

    def eval(self):
        """
        Evaluation Function
        :return: the overall averaged value of Recall
        """
        return np.average(
            [Recall.__user_recall(u_r, self.cutoff, self.relevant_items[u])
             for u, u_r in self.recommendations.items()]
        )
