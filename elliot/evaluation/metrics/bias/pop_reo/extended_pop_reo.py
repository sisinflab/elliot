"""
This is the implementation of the Popularity-based Ranking-based Equal Opportunity (REO) metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Alejandro Bellogín'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, alejandro.bellogin@uam.es'

import numpy as np

from elliot.evaluation.metrics.base_metric import BaseMetric


class ExtendedPopREO(BaseMetric):
    """
    Extended Popularity-based Ranking-based Equal Opportunity

    This class represents the implementation of the Extended Popularity-based Ranking-based Equal Opportunity (REO) recommendation metric.

    For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/3397271.3401177>`_

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        complex_metrics:
        - metric: ExtendedPopREO
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
        self._cutoff = self._evaluation_objects.cutoff
        self._relevance = self._evaluation_objects.relevance.binary_relevance

        self._pop_ratio = self._additional_data.get("pop_ratio", 0.8)
        self._pop_obj = self._evaluation_objects.pop.get_custom_pop_obj(self._pop_ratio)

        self._short_head = set(self._pop_obj.get_short_head())
        self._long_tail = set(self._pop_obj.get_long_tail())
        self._train = self._evaluation_objects.data.train_dict
        self._num = []
        self._den = []

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "ExtendedPopREO"

    def __user_pop_reo(self, user_recommendations, cutoff, long_tail, short_head, u_train, user_relevant_items):
        """
        Per User Popularity-based Ranking-based Equal Opportunity (REO)
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Average Recommendation Popularity metric for the specific user
        """
        recommended_items = set([i for i, _ in user_recommendations[:cutoff] if i in user_relevant_items])
        num_h = len(recommended_items & short_head)
        num_t = len(recommended_items & long_tail)
        den_h = len((short_head & user_relevant_items)-u_train)
        den_t = len((long_tail & user_relevant_items)-u_train)
        return num_h, num_t, den_h, den_t

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of PopREO
        """
        for u, u_r in self._recommendations.items():
            if len(self._relevance.get_user_rel(u)):
                num_h, num_t, den_h, den_t = self.__user_pop_reo(u_r, self._cutoff, self._long_tail, self._short_head, set(self._train[u].keys()), set(self._relevance.get_user_rel(u)))
                self._num.append([num_h, num_t])
                self._den.append([den_h, den_t])
        self._num = np.sum(np.array(self._num), axis=0)
        self._den = np.sum(np.array(self._den), axis=0)
        pr = self._num / self._den
        return np.std(pr)/np.mean(pr)

