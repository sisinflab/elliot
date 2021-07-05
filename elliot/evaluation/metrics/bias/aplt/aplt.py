"""
This is the implementation of the Average percentage of long tail items metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import operator

import numpy as np
from elliot.evaluation.metrics.base_metric import BaseMetric


class APLT(BaseMetric):
    r"""
    Average percentage of long tail items

    This class represents the implementation of the Average percentage of long tail items recommendation metric.

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3109859.3109912>`_

    .. math::
        \mathrm {ACLT}=\frac{1}{\left|U_{t}\right|} \sum_{u \in U_{t}} \frac{|\{i, i \in(L(u) \cap \sim \Phi)\}|}{|L(u)|}

    :math:`U_{t}` is the number of users in the test set.

    :math:`L_{u}` is the recommended list of items for user u.

    :math:`\sim \Phi`   medium-tail items.

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [APLT]
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
        self._long_tail = self._evaluation_objects.pop.get_long_tail()

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "APLT"

    @staticmethod
    def __user_aplt(user_recommendations, cutoff, long_tail):
        """
        Per User Average percentage of long tail items
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Average Recommendation Popularity metric for the specific user
        """
        return len(set([i for i,v in user_recommendations[:cutoff]]) & set(long_tail)) / len(user_recommendations[:cutoff])

    # def eval(self):
    #     """
    #     Evaluation function
    #     :return: the overall averaged value of APLT
    #     """
    #     return np.average(
    #         [APLT.__user_aplt(u_r, self._cutoff, self._long_tail)
    #          for u, u_r in self._recommendations.items()]
    #     )

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of APLT
        """
        return {u: APLT.__user_aplt(u_r, self._cutoff, self._long_tail)
             for u, u_r in self._recommendations.items()}

