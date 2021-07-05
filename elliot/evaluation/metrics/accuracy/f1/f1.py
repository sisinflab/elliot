"""
This is the implementation of the F-score metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from elliot.evaluation.metrics.base_metric import BaseMetric


class F1(BaseMetric):
    r"""
    F-Measure

    This class represents the implementation of the F-score recommendation metric.
    Passing 'F1' to the metrics list will enable the computation of the metric.

    For further details, please refer to the `paper <https://link.springer.com/chapter/10.1007/978-1-4899-7637-6_8>`_

    .. math::
        \mathrm {F1@K} = \frac{1+\beta^{2}}{\frac{1}{\text { precision@k }}+\frac{\beta^{2}}{\text { recall@k }}}

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [F1]
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
        self._beta = 1 # F-score is the Sørensen-Dice (DSC) coefficient with beta equal to 1
        self._squared_beta = self._beta**2

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "F1"

    @staticmethod
    def __user_f1(user_recommendations, cutoff, user_relevant_items, squared_beta):
        """
        Per User F-score
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        p = sum([1 for i in user_recommendations[:cutoff] if i[0] in user_relevant_items]) / cutoff
        r = sum([1 for i in user_recommendations[:cutoff] if i[0] in user_relevant_items]) / len(user_relevant_items)
        num = (1 + squared_beta) * p * r
        den = (squared_beta * p) + r
        return num/den if den != 0 else 0

    # def eval(self):
    #     """
    #     Evaluation function
    #     :return: the overall averaged value of F-score
    #     """
    #     return np.average(
    #         [F1.__user_f1(u_r, self._cutoff, self._relevant_items[u], self._squared_beta)
    #          for u, u_r in self._recommendations.items() if len(self._relevant_items[u])]
    #     )

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of F-score
        """
        return {u: F1.__user_f1(u_r, self._cutoff, self._relevance.get_user_rel(u), self._squared_beta)
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))}

