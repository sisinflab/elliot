"""
This is the implementation of the Shannon Entropy metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import math

from elliot.evaluation.metrics.base_metric import BaseMetric


class ShannonEntropy(BaseMetric):
    r"""
    Shannon Entropy

    This class represents the implementation of the Shannon Entropy recommendation metric.

    For further details, please refer to the `book <https://link.springer.com/10.1007/978-1-4939-7131-2_110158>`_

    .. math::
        \mathrm {ShannonEntropy}=-\sum_{i=1}^{n} p(i) \log p(i)

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [SEntropy]
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
        self._num_items = self._evaluation_objects.num_items
        self._item_count = {}
        self._item_weights = {}
        self._free_norm = 0
        self._ln2 = math.log(2.0)

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "SEntropy"

    def __user_se(self, user_recommendations, cutoff):
        """
        Per User computation useful for Shannon Entropy
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        user_norm = len(user_recommendations[:cutoff])
        self._free_norm += user_norm
        for i, _ in user_recommendations[:cutoff]:
            self._item_count[i] = self._item_count.get(i, 0) + 1
            self._item_weights[i] = self._item_weights.get(i, 0) + (1 / user_norm)

    def __sales_novelty(self, i):
        return -math.log(self._item_count[i] / self._free_norm) / self._ln2

    def eval(self):
        """
        Evaluation function
        :return: the overall value of Shannon Entropy
        """

        for u, u_r in self._recommendations.items():
            self.__user_se(u_r, self._cutoff)

        return sum([w * self.__sales_novelty(i) for i, w in self._item_weights.items()])/len(self._recommendations)


