"""
This is the implementation of the F-score metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Alejandro Bellogín'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, alejandro.bellogin@uam.es'

import importlib
import numpy as np
from elliot.evaluation.metrics.base_metric import BaseMetric
from elliot.evaluation.metrics.metrics_utils import ProxyStatisticalMetric
# import elliot.evaluation.metrics as metrics


class ExtendedF1(BaseMetric):
    r"""
    Extended F-Measure

    This class represents the implementation of the F-score recommendation metric.
    Passing 'ExtendedF1' to the metrics list will enable the computation of the metric.

    "Evaluating Recommender Systems" Gunawardana, Asela and Shani, Guy, In Recommender systems handbook pages 265--308, 2015

    For further details, please refer to the `paper <https://link.springer.com/chapter/10.1007/978-1-4899-7637-6_8>`_

    .. math::
        \mathrm {ExtendedF1@K} =\frac{2}{\frac{1}{\text { metric_0@k }}+\frac{1}{\text { metric_1@k }}}

    Args:
        metric_0: First considered metric (default: Precision)
        metric_1: Second considered metric (default: Recall)

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        complex_metrics:
        - metric: ExtendedF1
          metric_0: Precision
          metric_1: Recall

    """

    def __init__(self, recommendations, config, params, eval_objects, additional_data):
        """
        Constructor
        :param recommendations: list of recommendations in the form {user: [(item1,value1),...]}
        :param config: SimpleNameSpace that represents the configuration of the experiment
        :param params: Parameters of the model
        :param eval_objects: list of objects that may be useful for the computation of the different metrics
        """
        super().__init__(recommendations, config, params, eval_objects, additional_data)

        self._beta = 1 # F-score is the Sørensen-Dice (DSC) coefficient with beta equal to 1
        self._squared_beta = self._beta**2

        parse_metric_func = importlib.import_module("elliot.evaluation.metrics").parse_metric

        self._metric_0 = self._additional_data.get("metric_0", False)
        self._metric_1 = self._additional_data.get("metric_1", False)
        if self._metric_0 and self._metric_1:
            self._metric_0 = parse_metric_func(self._metric_0)(recommendations, config, params, eval_objects)
            self._metric_1 = parse_metric_func(self._metric_1)(recommendations, config, params, eval_objects)

        self.process()

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "ExtendedF1"

    @staticmethod
    def __user_f1(metric_0_value, metric_1_value, squared_beta):
        """
        Per User F-score
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        num = (1 + squared_beta) * metric_0_value * metric_1_value
        den = (squared_beta * metric_0_value) + metric_1_value
        return num/den if den != 0 else 0

    # def eval(self):
    #     """
    #     Evaluation function
    #     :return: the overall averaged value of F-score
    #     """
    #     return np.average(
    #         [F1.__user_f1(u_r, self._cutoff, self._relevant_items[u], self._squared_beta)
    #          for u, u_r in self._recommendations.items() if len(self._relevant_items[u])]
    #     )

    def eval_user_metric(self):
        pass

    def process(self):
        """
        Evaluation function
        :return: the overall value of Bias Disparity
        """

        metric_0_res = self._metric_0.eval_user_metric()
        metric_1_res = self._metric_1.eval_user_metric()

        user_val = {u: ExtendedF1.__user_f1(metric_0_res.get(u), metric_1_res.get(u), self._squared_beta)
                    for u in (set(metric_0_res.keys()) and set(metric_1_res.keys()))}

        val = np.average(list(user_val.values()))

        self._metric_objs_list = []
        self._metric_objs_list.append(ProxyStatisticalMetric(
            name=f"ExtendedF1_m0:{self._metric_0.name()}-m1:{self._metric_1.name()}",
            val=val,
            user_val=user_val,
            needs_full_recommendations=False))

    def get(self):
        return self._metric_objs_list
