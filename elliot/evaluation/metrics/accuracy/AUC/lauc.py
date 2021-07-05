"""
This is the implementation of the Limited AUC metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from elliot.evaluation.metrics.base_metric import BaseMetric
from elliot.utils import logging
import logging as pylog

class LAUC(BaseMetric):
    r"""
    Limited Area Under the Curve

    This class represents the implementation of the Limited AUC recommendation metric.
    Passing 'LAUC' to the metrics list will enable the computation of the metric.

    "Setting Goals and Choosing Metrics for Recommender System Evaluations" by Gunnar Schröder, et al.

    For further details, please refer to the `paper <https://wiki.epfl.ch/edicpublic/documents/Candidacy%20exam/Evaluation.pdf>`_


    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [LAUC]

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
        self.logger = logging.get_logger("Evaluator",  pylog.CRITICAL if config.config_test else pylog.DEBUG)
        self._cutoff = self._evaluation_objects.cutoff
        self._relevance = self._evaluation_objects.relevance.binary_relevance
        self._num_items = self._evaluation_objects.num_items

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "LAUC"

    @staticmethod
    def __user_auc_at_k(user_recommendations, cutoff, user_relevant_items, num_items, train_size):
        """
        Per User Limited AUC
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        neg_num = num_items - train_size - len(user_relevant_items) + 1
        pos_ranks = [r for r, (i, _) in enumerate(user_recommendations[:cutoff]) if i in user_relevant_items]
        return sum([(neg_num - r_r + p_r)/(neg_num) for p_r, r_r in enumerate(pos_ranks)])/min(cutoff, len(user_relevant_items))

    # def eval(self):
    #     """
    #     Evaluation function
    #     :return: the overall averaged value of LAUC
    #     """
    #
    #     return np.average(
    #         [LAUC.__user_auc_at_k(u_r, self._cutoff, self._relevant_items[u], self._num_items, len(self._evaluation_objects.data.train_dict[u]))
    #          for u, u_r in self._recommendations.items() if len(self._relevant_items[u])]
    #     )

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of LAUC per user
        """
        return {u: LAUC.__user_auc_at_k(u_r, self._cutoff, self._relevance.get_user_rel(u), self._num_items, len(self._evaluation_objects.data.train_dict[u]))
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))}

