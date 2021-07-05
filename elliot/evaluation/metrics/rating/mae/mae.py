"""
This is the implementation of the Mean Absolute Error metric.
It proceeds from a system-wise computation.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from elliot.evaluation.metrics.base_metric import BaseMetric
from elliot.utils import logging


class MAE(BaseMetric):
    r"""
    Mean Absolute Error

    This class represents the implementation of the Mean Absolute Error recommendation metric.

    For further details, please refer to the `link <https://en.wikipedia.org/wiki/Mean_absolute_error>`_

    .. math::
        \mathrm{MAE}=\frac{1}{|{T}|} \sum_{(u, i) \in {T}}\left|\hat{r}_{u i}-r_{u i}\right|

    :math:`T` is the test set, :math:`\hat{r}_{u i}` is the score predicted by the model,

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [MAE]
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
        self._relevance = self._evaluation_objects.relevance.binary_relevance
        self._total_relevant_items = sum([len(self._relevance.get_user_rel(u)) for u, _ in self._recommendations.items()])
        self._test = self._evaluation_objects.relevance.get_test()

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "MAE"

    @staticmethod
    def __user_MAE(user_recommendations, user_test, user_relevant_items):
        """
        Per User computation for Mean Absolute Error
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        return sum([abs(v - user_test[i]) for i, v in user_recommendations if i in user_relevant_items])

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of Mean Absolute Error
        """
        return sum(
            [MAE.__user_MAE(u_r, self._test[u], self._relevance.get_user_rel(u))
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))]
        ) / self._total_relevant_items

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of Mean Absolute Error per user
        """
        return {u: MAE.__user_MAE(u_r, self._test[u], self._relevance.get_user_rel(u))/len(self._relevance.get_user_rel(u))
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))}

    @staticmethod
    def needs_full_recommendations():
        _logger = logging.get_logger("Evaluator")
        _logger.warn("WARNING: Mean Absolute Error metric requires full length recommendations")
        return True
