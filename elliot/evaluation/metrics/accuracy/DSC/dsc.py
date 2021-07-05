"""
This is the implementation of the Sørensen–Dice coefficient metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Alejandro Bellogín'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, alejandro.bellogin@uam.es'

import importlib
from elliot.evaluation.metrics.base_metric import BaseMetric
# import elliot.evaluation.metrics as metrics


class DSC(BaseMetric):
    r"""
    Sørensen–Dice coefficient

    This class represents the implementation of the Sørensen–Dice coefficient recommendation metric.
    Passing 'DSC' to the metrics list will enable the computation of the metric.

    For further details, please refer to the `page <https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient>`_

    .. math::
        \mathrm {DSC@K} = \frac{1+\beta^{2}}{\frac{1}{\text { metric_0@k }}+\frac{\beta^{2}}{\text { metric_1@k }}}

    Args:
        beta: the beta coefficient (default: 1)
        metric_0: First considered metric (default: Precision)
        metric_1: Second considered metric (default: Recall)

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        complex_metrics:
        - metric: DSC
          beta: 1
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

        self._beta = self._additional_data.get("beta", 1)
        self._squared_beta = self._beta**2

        metric_lib = importlib.import_module("elliot.evaluation.metrics")

        self._metric_0 = self._additional_data.get("metric_0", False)
        self._metric_1 = self._additional_data.get("metric_1", False)

        if self._metric_0:
            self._metric_0 = metric_lib.parse_metric(self._metric_0)(recommendations, config, params, eval_objects)
        else:
            self._metric_0 = metric_lib.Precision(recommendations, config, params, eval_objects)

        if self._metric_1:
            self._metric_1 = metric_lib.parse_metric(self._metric_1)(recommendations, config, params, eval_objects)
        else:
            self._metric_1 = metric_lib.Recall(recommendations, config, params, eval_objects)

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "DSC"

    @staticmethod
    def __user_dsc(metric_0_value, metric_1_value, squared_beta):
        """
        Per User Sørensen–Dice coefficient
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        num = (1 + squared_beta) * metric_0_value * metric_1_value
        den = (squared_beta * metric_0_value) + metric_1_value
        return num / den if den != 0 else 0

    # def eval(self):
    #     """
    #     Evaluation function
    #     :return: the overall averaged value of Sørensen–Dice coefficient
    #     """
    #     return np.average(
    #         [DSC.__user_dsc(u_r, self._cutoff, self._relevant_items[u], self._squared_beta)
    #          for u, u_r in self._recommendations.items() if len(self._relevant_items[u])]
    #     )

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of Sørensen–Dice coefficient per user
        """

        metric_0_res = self._metric_0.eval_user_metric()
        metric_1_res = self._metric_1.eval_user_metric()

        return {u: DSC.__user_dsc(metric_0_res.get(u), metric_1_res.get(u), self._squared_beta)
                    for u in (set(metric_0_res.keys()) and set(metric_1_res.keys()))}

