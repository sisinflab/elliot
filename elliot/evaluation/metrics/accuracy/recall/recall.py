"""
This is the implementation of the Recall metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from elliot.evaluation.metrics.base_metric import BaseMetric


class Recall(BaseMetric):
    r"""
    Recall-measure

    This class represents the implementation of the Recall recommendation metric.

    For further details, please refer to the `link <https://en.wikipedia.org/wiki/Precision_and_recall#Recall>`_

    .. math::
        \mathrm {Recall@K} = \frac{|Rel_u\cap Rec_u|}{Rel_u}

    :math:`Rel_u` is the set of items relevant to user :math:`U`,

    :math:`Rec_u` is the top K items recommended to users.

    We obtain the result by calculating the average :math:`Recall@K` of each user.

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [Recall]
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
        self._relevance = self._evaluation_objects.relevance.binary_relevance

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "Recall"

    def __user_recall(self, user_recommendations, user, cutoff):
        """
        Per User Recall
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Recall metric for the specific user
        """
        return sum([self._relevance.get_rel(user, i) for i, _ in user_recommendations[:cutoff]]) / len(self._relevance.get_user_rel(user))

    # def eval(self):
    #     """
    #     Evaluation Function
    #     :return: the overall averaged value of Recall
    #     """
    #     return np.average(
    #         [Recall.__user_recall(u_r, self._cutoff, self._relevant_items[u])
    #          for u, u_r in self._recommendations.items() if len(self._relevant_items[u])]
    #     )

    def eval_user_metric(self):
        """
        Evaluation Function
        :return: the overall averaged value of Recall per user
        """
        return {u: self.__user_recall(u_r, u, self._cutoff)
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))}
