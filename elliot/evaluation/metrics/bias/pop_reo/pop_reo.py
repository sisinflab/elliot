"""
This is the implementation of the Popularity-based Ranking-based Equal Opportunity (REO) metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np

from elliot.evaluation.metrics.base_metric import BaseMetric


class PopREO(BaseMetric):
    r"""
    Popularity-based Ranking-based Equal Opportunity

    This class represents the implementation of the Popularity-based Ranking-based Equal Opportunity (REO) recommendation metric.

    For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/3397271.3401177>`_

    .. math::
        \mathrm {REO}=\frac{{std}\left(P\left(R @ k \mid g=g_{1}, y=1\right) \ldots P\left(R(a) k=g_{A}, y=1\right)\right)}
        {{mean}\left(P\left(R @ k \mid g=g_{1}, y=1\right) \ldots P\left(R @ k \mid g=g_{A}, y=1\right)\right)}

    :math:`P\left(R @ k \mid g=g_{a}, y=1\right) = \frac{\sum_{u=1}^{N} \sum_{i=1}^{k} G_{g_{a}}\left(R_{u, i}\right) Y\left(u, R_{u, i}\right)}
    {\sum_{u=1}^{N} \sum_{i \in I \backslash I_{u}^{+}} G_{g_{a}}(i) Y(u, i)}`

    :math:`Y\left(u, R_{u, i}\right)` identifies the ground-truth label of a user-item pair `\left(u, R_{u, i}\right)`,
    if item `R_{u, i}` is liked by user 𝑢, returns 1, otherwise 0

    :math:`\sum_{i=1}^{k} G_{g_{a}}\left(R_{u, i}\right) Y\left(u, R_{u, i}\right)`
    counts how many items in test set from group `{g_a}` are ranked in top-𝑘 for user u

    :math:`\sum_{i \in I \backslash I_{u}^{+}} G_{g_{a}}(i) Y(u, i)`
    counts the total number of items from group `{g_a}` 𝑎 in test set for user u

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [PopREO]
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
        self._short_head = set(self._evaluation_objects.pop.get_short_head())
        self._long_tail = set(self._evaluation_objects.pop.get_long_tail())
        self._train = self._evaluation_objects.data.train_dict
        self._num = []
        self._den = []

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "PopREO"

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

