"""
This is the implementation of the Precision metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from ..base_metric import BaseMetric


class AUC(BaseMetric):
    """
    This class represents the implementation of the Precision recommendation metric.
    Passing 'Precision' to the metrics list will enable the computation of the metric.
    """

    def __init__(self, recommendations, config, params, eval_objects):
        """
        Constructor
        :param recommendations: list of recommendations in the form {user: [(item1,value1),...]}
        :param cutoff: numerical threshold to limit the recommendation list
        :param relevant_items: list of relevant items (binary) per user in the form {user: [item1,...]}
        """
        super().__init__(recommendations, config, params, eval_objects)
        self._cutoff = self._evaluation_objects.cutoff
        self._relevant_items = self._evaluation_objects.relevance.get_binary_relevance()
        self._num_items = self._evaluation_objects.num_items

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "AUC"

    @staticmethod
    def __user_auc(user_recommendations, user_relevant_items, num_items, train_size):
        """
        Per User Precision
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        neg_num = num_items - train_size

        return [(neg_num - r)/neg_num for r, (i, _) in enumerate(user_recommendations) if i in user_relevant_items]

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of Precision
        """
        list_of_lists = [AUC.__user_auc(u_r, self._relevant_items[u], self._num_items, len(self._evaluation_objects.data.train_dict[u]))
             for u, u_r in self._recommendations.items()]
        return np.average([item for sublist in list_of_lists for item in sublist])

    @staticmethod
    def needs_full_recommendations():
        return True