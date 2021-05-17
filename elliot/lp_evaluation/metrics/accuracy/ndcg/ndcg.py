"""
This is the implementation of the normalized Discounted Cumulative Gain metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Alejandro Bellogín'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, alejandro.bellogin@uam.es'

import typing as t

import numpy as np

from elliot.lp_evaluation.metrics.base_metric import BaseMetric


class nDCG(BaseMetric):
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

    def __init__(self, ranks_l: t.List[int], ranks_r: t.List[int], config, params, eval_objects):
        """
        Constructor
        :param recommendations: list of recommendations in the form {user: [(item1,value1),...]}
        :param config: SimpleNameSpace that represents the configuration of the experiment
        :param params: Parameters of the model
        :param eval_objects: list of objects that may be useful for the computation of the different metrics
        """
        super().__init__(ranks_l, ranks_r, config, params, eval_objects)
        self._cutoff = self._evaluation_objects.cutoff

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "nDCG"

    def __triple_ndcg(self, rank: int, cutoff: int):
        """
        Per User normalized Discounted Cumulative Gain
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param user_gain_map: dict of discounted relevant items in the form {user1:{item1:value1,...},...}
        :param cutoff: numerical threshold to limit the recommendation list
        :return: the value of the nDCG metric for the specific user
        """
        return 1 / (np.log(rank + 1) / np.log(2)) if rank <= cutoff else 0

    def eval_triple_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of normalized Discounted Cumulative Gain per user
        """

        return [self.__triple_ndcg(rank, self._cutoff) for rank in self._ranks_l + self._ranks_r]





