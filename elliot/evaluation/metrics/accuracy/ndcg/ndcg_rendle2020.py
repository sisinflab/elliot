"""
This is the implementation of the normalized Discounted Cumulative Gain metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Alejandro Bellogín'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, alejandro.bellogin@uam.es'

import typing as t

import numpy as np

from elliot.evaluation.metrics.base_metric import BaseMetric


class nDCGRendle2020(BaseMetric):
    r"""
    normalized Discounted Cumulative Gain

    This class represents the implementation of the nDCG recommendation metric.

    For further details, please refer to the `link <https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG>`_

    .. math::
        \begin{gather}
            \mathrm {DCG@K}=\sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_{2}{(i+1)}}\\
            \mathrm {IDCG@K}=\sum_{i=1}^{K}\frac{1}{\log_{2}{(i+1)}}\\
            \mathrm {NDCG_u@K}=\frac{DCG_u@K}{IDCG_u@K}\\
            \mathrm {NDCG@K}=\frac{\sum \nolimits_{u \in u^{te}NDCG_u@K}}{|u^{te}|}
        \end{gather}


    :math:`K` stands for recommending :math:`K` items.

    And the :math:`rel_i` is the relevance of the item in position :math:`i` in the recommendation list.

    :math:`2^{rel_i}` equals to 1 if the item hits otherwise 0.

    :math:`U^{te}` is for all users in the test set.

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [nDCG]
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
        self._rel_threshold = self._evaluation_objects.relevance._rel_threshold

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "nDCGRendle2020"

    def __user_ndcg(self, user_recommendations: t.List, user, cutoff: int):
        """
        Per User normalized Discounted Cumulative Gain
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param user_gain_map: dict of discounted relevant items in the form {user1:{item1:value1,...},...}
        :param cutoff: numerical threshold to limit the recommendation list
        :return: the value of the nDCG metric for the specific user
        """

        idcg: float = 1 / (sum([1 / (np.log(i+2) / np.log(2)) for i in range(min(len(self._relevance.get_user_rel(user)), cutoff))]))
        return idcg * sum([1 / (np.log(p+2) / np.log(2)) for p, (i, _) in enumerate(user_recommendations[:cutoff]) if self._relevance.get_rel(user, i)])

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of normalized Discounted Cumulative Gain per user
        """

        return {u: self.__user_ndcg(u_r, u, self._cutoff)
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))}





