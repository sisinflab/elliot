"""
This is the implementation of the Average Recommendation Popularity metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from elliot.evaluation.metrics.base_metric import BaseMetric


class ARP(BaseMetric):
    r"""
    Average Recommendation Popularity

    This class represents the implementation of the Average Recommendation Popularity recommendation metric.

    For further details, please refer to the `paper <https://arxiv.org/abs/1205.6700>`_

    .. math::
        \mathrm {ARP}=\frac{1}{\left|U_{t}\right|} \sum_{u \in U_{t}} \frac{\sum_{i \in L_{u}} \phi(i)}{\left|L_{u}\right|}

    :math:`U_{t}` is the number of users in the test set.

    :math:`L_{u}` is the recommended list of items for user u.

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [ARP]
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
        self._pop_items = self._evaluation_objects.pop.get_pop_items()

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "ARP"

    @staticmethod
    def __user_arp(user_recommendations, cutoff, pop_items):
        """
        Per User Average Recommendation Popularity
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Average Recommendation Popularity metric for the specific user
        """
        return sum([pop_items[i] for i, v in user_recommendations[:cutoff]]) / len(user_recommendations[:cutoff])

    # def eval(self):
    #     """
    #     Evaluation function
    #     :return: the overall averaged value of ARP
    #     """
    #     return np.average(
    #         [ARP.__user_arp(u_r, self._cutoff, self._pop_items)
    #          for u, u_r in self._recommendations.items()]
    #     )

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of ARP
        """
        return {u: ARP.__user_arp(u_r, self._cutoff, self._pop_items)
             for u, u_r in self._recommendations.items()}

