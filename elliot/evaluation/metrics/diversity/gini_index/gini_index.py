"""
This is the implementation of the Gini Index metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from elliot.evaluation.metrics.base_metric import BaseMetric


class GiniIndex(BaseMetric):
    r"""
    Gini Index

    This class represents the implementation of the Gini Index recommendation metric.

    For further details, please refer to the `book <https://link.springer.com/10.1007/978-1-4939-7131-2_110158>`_

    .. math::
        \mathrm {GiniIndex}=\frac{1}{n-1} \sum_{j=1}^{n}(2 j-n-1) p\left(i_{j}\right)

    :math:`i_{j}` is the list of items ordered according to increasing `p(i)`

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [Gini]
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
        self._free_norm = 0

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "Gini"

    def __user_gini(self, user_recommendations, cutoff):
        """
        Per User Gini Index
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        user_norm = len(user_recommendations[:cutoff])
        self._free_norm += user_norm
        for i, _ in user_recommendations[:cutoff]:
            self._item_count[i] = self._item_count.get(i, 0) + 1

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of Gini Index
        """

        for u, u_r in self._recommendations.items():
            self.__user_gini(u_r, self._cutoff)

        n_recommended_items = len(self._item_count)

        gini = sum([(2 * (j + (self._num_items - n_recommended_items) + 1) - self._num_items - 1) * (cs / self._free_norm) for j, cs in enumerate(sorted(self._item_count.values()))])
        gini /= (self._num_items - 1)
        gini = 1 - gini

        return gini


