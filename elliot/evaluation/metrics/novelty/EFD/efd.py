"""
This is the implementation of the Precision metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import math
import numpy as np
from evaluation.metrics.base_metric import BaseMetric


class EFD(BaseMetric):
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

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "EFD"

    def __user_EFD(self, user_recommendations, cutoff, user_relevant_items):
        """
        Per User EFD
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """

        nov = 0
        norm = 0
        for r, (i, _) in enumerate(user_recommendations[:cutoff]):
            if i in user_relevant_items:
                nov += EFD.__discount_k(r) * self._item_novelty_dict.get(i, self._max_nov)
            norm += EFD.__discount_k(r)

        if norm > 0:
            nov /= norm

        return nov

    @staticmethod
    def __discount_k(k):
        return (1 / math.log(k + 2)) * math.log(2)

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of Precision
        """

        self._item_count = {}
        for u_h in self._evaluation_objects.data.train_dict.values():
            for i in u_h.keys():
                self._item_count[i] = self._item_count.get(i, 0) + 1

        novelty_profile = self._item_count.values()
        norm = sum(novelty_profile)
        self._max_nov = -math.log(min(novelty_profile) / norm) / math.log(2)
        self._item_novelty_dict = {i: -math.log(v / norm) / math.log(2) for i, v in self._item_count.items()}

        return np.average([self.__user_EFD(u_r, self._cutoff, self._relevant_items[u])
             for u, u_r in self._recommendations.items()])

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of Precision
        """

        self._item_count = {}
        for u_h in self._evaluation_objects.data.train_dict.values():
            for i in u_h.keys():
                self._item_count[i] = self._item_count.get(i, 0) + 1

        novelty_profile = self._item_count.values()
        norm = sum(novelty_profile)
        self._max_nov = -math.log(min(novelty_profile) / norm) / math.log(2)
        self._item_novelty_dict = {i: -math.log(v / norm) / math.log(2) for i, v in self._item_count.items()}

        return {u: self.__user_EFD(u_r, self._cutoff, self._relevant_items[u])
                for u, u_r in self._recommendations.items()}

