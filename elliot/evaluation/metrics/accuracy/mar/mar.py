"""
This is the implementation of the Mean Average Recall metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from elliot.evaluation.metrics.base_metric import BaseMetric


class MAR(BaseMetric):
    r"""
    Mean Average Recall

    This class represents the implementation of the Mean Average Recall recommendation metric.
    Passing 'MAR' to the metrics list will enable the computation of the metric.

    For further details, please refer to the `link <http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#So-Why-Did-I-Bother-Defining-Recall?>`_

    .. math::
        \begin{align*}
        \mathrm{Recall@N} &= \frac{1}{\mathrm{min}(m,|rel(k)|)}\sum_{k=1}^N P(k) \cdot rel(k) \\
        \mathrm{MAR@N}& = \frac{1}{|U|}\sum_{u=1}^{|U|}(\mathrm{Recall@N})_u
        \end{align*}

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [MAR]
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

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "MAR"

    @staticmethod
    def __user_ar(user_recommendations, cutoff, user_relevant_items):
        """
        Per User Average Recall
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Recall metric for the specific user
        """
        return np.average([MAR.__user_recall(user_recommendations[:cutoff], (n+1), user_relevant_items) for n in range(cutoff)])

    @staticmethod
    def __user_recall(user_recommendations, cutoff, user_relevant_items):
        """
        Per User Recall
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Recall metric for the specific user
        """
        return sum([1 for i in user_recommendations[:cutoff] if i[0] in user_relevant_items]) / len(user_relevant_items)

    # def eval(self):
    #     """
    #     Evaluation function
    #     :return: the overall averaged value of Mean Average Recall
    #     """
    #     return np.average(
    #         [MAR.__user_ar(u_r, self._cutoff, self._relevant_items[u])
    #          for u, u_r in self._recommendations.items() if len(self._relevant_items[u])]
    #     )

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of Mean Average Recall per user
        """
        return {u: MAR.__user_ar(u_r, self._cutoff, self._relevance.get_user_rel(u))
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))}

