"""
This is the implementation of the global AUC metric.
It proceeds from a system-wise computation.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np

from elliot.evaluation.metrics.base_metric import BaseMetric
from elliot.utils import logging


class AUC(BaseMetric):
    r"""
    Area Under the Curve

    This class represents the implementation of the global AUC recommendation metric.
    Passing 'AUC' to the metrics list will enable the computation of the metric.

    For further details, please refer to the `AUC <https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve>`_

    Note:
        This metric does not calculate group-based AUC which considers the AUC scores
        averaged across users. It is also not limited to k. Instead, it calculates the
        scores on the entire prediction results regardless the users.

    .. math::
        \mathrm {AUC} = \frac{\sum\limits_{i=1}^M rank_{i}
        - \frac {{M} \times {(M+1)}}{2}} {{{M} \times {N}}}

    :math:`M` is the number of positive samples.

    :math:`N` is the number of negative samples.

    :math:`rank_i` is the ascending rank of the ith positive sample.

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [AUC]
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
        self._num_items = self._evaluation_objects.num_items

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "AUC"

    @staticmethod
    def __user_auc(user_recommendations, user_relevant_items, num_items, train_size):
        """
        Per User Computation of AUC values
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :param num_items: overall number of items considered in the training set
        :param train_size: length of the user profile
        :return: the list of the AUC values per each test item
        """
        neg_num = num_items - train_size - len(user_relevant_items) + 1
        pos_ranks = [r for r, (i, _) in enumerate(user_recommendations) if i in user_relevant_items]
        return [(neg_num - r_r + p_r) / (neg_num) for p_r, r_r in enumerate(pos_ranks)]

    def eval(self):
        """
        Evaluation function
        :return: the overall value of AUC
        """
        list_of_lists = [AUC.__user_auc(u_r, self._relevance.get_user_rel(u), self._num_items, len(self._evaluation_objects.data.train_dict[u]))
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))]
        return np.average([item for sublist in list_of_lists for item in sublist])

    @staticmethod
    def needs_full_recommendations():
        _logger = logging.get_logger("Evaluator")
        _logger.warn("AUC metric requires full length recommendations")
        return True
