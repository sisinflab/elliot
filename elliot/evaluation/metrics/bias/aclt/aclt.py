"""
This is the implementation of the Average coverage of long tail items metric.
It proceeds from a user-wise computation, and average the values over the users.
"""


import numpy as np
from elliot.evaluation.metrics.base_metric import BaseMetric
from elliot.utils.registry import metric_registry


@metric_registry.register()
class ACLT(BaseMetric):
    r"""
    Average coverage of long tail items

    This class represents the implementation of the Average coverage of long tail items recommendation metric.

    For further details, please refer to the `paper <https://arxiv.org/abs/1901.07555>`_

    .. math::
        \mathrm {ACLT}=\frac{1}{\left|U_{t}\right|} \sum_{u \in U_{f}} \sum_{i \in L_{u}} 1(i \in \Gamma)

    :math:`U_{t}` is the number of users in the test set.

    :math:`L_{u}` is the recommended list of items for user u.

    :math:`1(i \in \Gamma)`  is an indicator function and it equals to 1 when i is in \Gamma.

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [ACLT]
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
        pop_cache = getattr(self._evaluation_objects, "pop_cache", None)
        self._long_tail = (
            pop_cache.long_tail_set
            if pop_cache and getattr(pop_cache, "long_tail_set", None) is not None
            else set(self._evaluation_objects.pop.get_long_tail())
        )

    @staticmethod
    def __user_aclt(user_recommendations, cutoff, long_tail):
        """
        Per User Average coverage of long tail items
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Average Recommendation Popularity metric for the specific user
        """
        return len({i for i, _ in user_recommendations[:cutoff]} & long_tail)

    # def eval(self):
    #     """
    #     Evaluation function
    #     :return: the overall averaged value of ACLT
    #     """
    #     return np.average(
    #         [ACLT.__user_aclt(u_r, self._cutoff, self._long_tail)
    #          for u, u_r in self._recommendations.items()]
    #     )

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of ACLT
        """
        return {u: ACLT.__user_aclt(u_r, self._cutoff, self._long_tail)
             for u, u_r in self._recommendations.items()}
