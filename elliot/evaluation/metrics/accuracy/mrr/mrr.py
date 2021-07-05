"""
This is the implementation of the Mean Reciprocal Rank metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from elliot.evaluation.metrics.base_metric import BaseMetric


class MRR(BaseMetric):
    r"""
    Mean Reciprocal Rank

    This class represents the implementation of the Mean Reciprocal Rank recommendation metric.
    Passing 'MRR' to the metrics list will enable the computation of the metric.

    For further details, please refer to the `link <https://en.wikipedia.org/wiki/Mean_reciprocal_rank>`_

    .. math::
        \mathrm {MRR} = \frac{1}{|{U}|} \sum_{i=1}^{|{U}|} \frac{1}{rank_i}
    :math:`U` is the number of users, :math:`rank_i` is the rank of the first item in the recommendation list
    in the test set results for user :math:`i`.

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [MRR]
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
        return "MRR"

    @staticmethod
    def __user_mrr(user_recommendations, cutoff, user_relevant_items):
        """
        Per User Mean Reciprocal Rank
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        return MRR.__get_reciprocal_rank(user_recommendations[:cutoff], user_relevant_items)

    @staticmethod
    def __get_reciprocal_rank(user_recommendations, user_relevant_items):
        for r, (i, v) in enumerate(user_recommendations):
            if i in user_relevant_items:
                return 1 / (r + 1)
        return 0

    # def eval(self):
    #     """
    #     Evaluation function
    #     :return: the overall averaged value of Mean Reciprocal Rank
    #     """
    #     return np.average(
    #         [MRR.__user_mrr(u_r, self._cutoff, self._relevant_items[u])
    #          for u, u_r in self._recommendations.items() if len(self._relevant_items[u])]
    #     )

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of Mean Reciprocal Rank per user
        """
        return {u: MRR.__user_mrr(u_r, self._cutoff, self._relevance.get_user_rel(u))
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))}

