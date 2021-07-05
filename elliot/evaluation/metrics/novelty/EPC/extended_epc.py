"""
This is the implementation of the Expected Popularity Complement metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import math
import numpy as np
from elliot.evaluation.metrics.base_metric import BaseMetric


class ExtendedEPC(BaseMetric):
    r"""
    Extended EPC

    This class represents the implementation of the Extended EPC recommendation metric.

    For further details, please refer to the `paper <https://dl.acm.org/doi/pdf/10.1145/2043932.2043955>`_

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        complex_metrics:
        - metric: ExtendedEPC
    """

    def __init__(self, recommendations, config, params, eval_objects, additional_data):
        """
        Constructor
        :param recommendations: list of recommendations in the form {user: [(item1,value1),...]}
        :param config: SimpleNameSpace that represents the configuration of the experiment
        :param params: Parameters of the model
        :param eval_objects: list of objects that may be useful for the computation of the different metrics
        """
        super().__init__(recommendations, config, params, eval_objects, additional_data)
        self._cutoff = self._evaluation_objects.cutoff
        self._relevance = self._evaluation_objects.relevance.binary_relevance

        self._relevance_type = self._additional_data.get("relevance", "binary")
        if self._relevance_type == "discounted":
            self._relevance = self._evaluation_objects.relevance.discounted_relevance
        else:
            self._relevance = self._evaluation_objects.relevance.binary_relevance

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "ExtendedEPC"

    def __user_EPC(self, user_recommendations, user, cutoff):
        """
        Per User Expected Popularity Complement
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """

        nov = 0
        norm = 0
        for r, (i, _) in enumerate(user_recommendations[:cutoff]):
            nov += self._relevance.get_rel(user, i) * self._relevance.logarithmic_ranking_discount(r) * self._item_novelty_dict.get(i, 1)
            norm += self._relevance.logarithmic_ranking_discount(r)

        if norm > 0:
            nov /= norm

        return nov

    # @staticmethod
    # def __discount_k(k):
    #     return (1 / math.log(k + 2)) * math.log(2)

    # def eval(self):
    #     """
    #     Evaluation function
    #     :return: the overall averaged value of Expected Popularity Complement
    #     """
    #
    #     item_count = {}
    #     for u_h in self._evaluation_objects.data.train_dict.values():
    #         for i in u_h.keys():
    #             item_count[i] = item_count.get(i, 0) + 1
    #
    #     num_users = len(self._evaluation_objects.data.train_dict)
    #     self._item_novelty_dict = {i: 1 - (v / num_users) for i, v in item_count.items()}
    #
    #     a = [self.__user_EPC(u_r, u, self._cutoff)
    #          for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))]
    #     return np.average(a)

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of Expected Popularity Complement per user
        """


        item_count = {}
        for u_h in self._evaluation_objects.data.train_dict.values():
            for i in u_h.keys():
                item_count[i] = item_count.get(i, 0) + 1

        num_users = len(self._evaluation_objects.data.train_dict)
        self._item_novelty_dict = {i: 1 - (v / num_users) for i, v in item_count.items()}

        return {u: self.__user_EPC(u_r, u, self._cutoff)
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))}

