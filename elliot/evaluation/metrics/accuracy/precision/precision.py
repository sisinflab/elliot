"""
This is the implementation of the Precision metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from elliot.evaluation.metrics.base_metric import BaseMetric


class Precision(BaseMetric):
    r"""
    Precision-measure

    This class represents the implementation of the Precision recommendation metric.

    For further details, please refer to the `link <https://en.wikipedia.org/wiki/Precision_and_recall#Precision>`_

    .. math::
        \mathrm {Precision@K} = \frac{|Rel_u \cap Rec_u|}{Rec_u}

    :math:`Rel_u` is the set of items relevant to user :math:`U`,

    :math:`Rec_u` is the top K items recommended to users.

    We obtain the result by calculating the average :math:`Precision@K` of each user.

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [Precision]
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
        return "Precision"

    def __user_precision(self, user_recommendations, user, cutoff):
        """
        Per User Precision
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        return sum([self._relevance.get_rel(user, i) for i, _ in user_recommendations[:cutoff]]) / cutoff

    # def eval(self):
    #     """
    #     Evaluation function
    #     :return: the overall averaged value of Precision
    #     """
    #     return np.average(
    #         [Precision.__user_precision(u_r, self._cutoff, self._relevant_items[u])
    #          for u, u_r in self._recommendations.items() if len(self._relevant_items[u])]
    #     )

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of Precision
        """
        return {u: self.__user_precision(u_r, u, self._cutoff)
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))}

