"""
This is the implementation of the Hit Rate metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import typing as t
from elliot.lp_evaluation.metrics.base_metric import BaseMetric


class HR(BaseMetric):
    r"""
    Hit Rate

    This class represents the implementation of the Hit Rate recommendation metric.
    Passing 'HR' to the metrics list will enable the computation of the metric.

    For further details, please refer to the `link <https://medium.com/@rishabhbhatia315/recommendation-system-evaluation-metrics-3f6739288870>`_

    .. math::
        \mathrm {HR@K} =\frac{Number \space of \space Hits @K}{|GT|}

    :math:`HR` is the number of users with a positive sample in the recommendation list.

    :math:`GT` is the total number of samples in the test set.

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [HR]
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
        return "HR"

    @staticmethod
    def __triple_HR(rank, cutoff):
        """
        Per User Hit Rate
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        return 1 if rank <= cutoff else 0

    def eval_triple_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of Hit Rate per user
        """
        return [HR.__triple_HR(rank, self._cutoff) for rank in self._ranks_l + self._ranks_r]

