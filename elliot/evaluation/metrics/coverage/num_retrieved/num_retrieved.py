"""
This is the implementation of the NumRetrieved metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from elliot.evaluation.metrics.base_metric import BaseMetric


class NumRetrieved(BaseMetric):
    r"""
    Number of Recommendations Retrieved

    This class represents the implementation of the Number of Recommendations Retrieved recommendation metric.

    For further details, please refer to the `link <https://github.com/RankSys/RankSys/blob/master/RankSys-metrics/src/main/java/es/uam/eps/ir/ranksys/metrics/basic/NumRetrieved.java>`_

    .. code:: yaml

        simple_metrics: [NumRetrieved]
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
        return "NumRetrieved"

    @staticmethod
    def __user_num_retrieved(user_recommendations, cutoff):
        """
        Per User NumRetrieved
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        return len(user_recommendations[:cutoff])

    # def eval(self):
    #     """
    #     Evaluation function
    #     :return: the overall averaged value of NumRetrieved
    #     """
    #     return np.average(
    #         [NumRetrieved.__user_num_retrieved(u_r, self._cutoff)
    #          for u, u_r in self._recommendations.items() if len(self._relevant_items[u])]
    #     )

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of NumRetrieved
        """
        return {u: NumRetrieved.__user_num_retrieved(u_r, self._cutoff)
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))}

