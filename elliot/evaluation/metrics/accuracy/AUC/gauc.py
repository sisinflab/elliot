"""
This is the implementation of the GroupAUC metric.
It proceeds from a user-wise computation, and average the AUC values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np

from elliot.evaluation.metrics.base_metric import BaseMetric
from elliot.utils import logging


class GAUC(BaseMetric):
    r"""
    Group Area Under the Curve

    This class represents the implementation of the GroupAUC recommendation metric.
    Passing 'GAUC' to the metrics list will enable the computation of the metric.

    "Deep Interest Network for Click-Through Rate Prediction" KDD '18 by Zhou, et al.

    For further details, please refer to the `paper <https://www.ijcai.org/Proceedings/2019/0319.pdf>`_

    Note:
        It calculates the AUC score of each user, and finally obtains GAUC by weighting the user AUC.
        It is also not limited to k. Due to our padding for `scores_tensor` in `RankEvaluator` with
        `-np.inf`, the padding value will influence the ranks of origin items. Therefore, we use
        descending sort here and make an identity transformation  to the formula of `AUC`, which is
        shown in `auc_` function. For readability, we didn't do simplification in the code.

    .. math::
        \mathrm {GAUC} = \frac {{{M} \times {(M+N+1)} - \frac{M \times (M+1)}{2}} -
        \sum\limits_{i=1}^M rank_{i}} {{M} \times {N}}

    :math:`M` is the number of positive samples.

    :math:`N` is the number of negative samples.

    :math:`rank_i` is the descending rank of the ith positive sample.

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [GAUC]
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
        return "GAUC"

    @staticmethod
    def __user_gauc(user_recommendations, user_relevant_items, num_items, train_size):
        """
        Per User AUC
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        neg_num = num_items - train_size - len(user_relevant_items) + 1
        pos_ranks = [r for r, (i, _) in enumerate(user_recommendations) if i in user_relevant_items]
        return sum([(neg_num - r_r + p_r)/(neg_num) for p_r, r_r in enumerate(pos_ranks)])/len(user_relevant_items)

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of AUC
        """

        return np.average(
            [GAUC.__user_gauc(u_r, self._relevance.get_user_rel(u), self._num_items, len(self._evaluation_objects.data.train_dict[u]))
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))]
        )

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of AUC per user
        """
        return {u: GAUC.__user_gauc(u_r, self._relevance.get_user_rel(u), self._num_items, len(self._evaluation_objects.data.train_dict[u]))
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))}


    @staticmethod
    def needs_full_recommendations():
        _logger = logging.get_logger("Evaluator")
        _logger.warn("\n*** WARNING: Group AUC metric requires full length recommendations")
        return True

