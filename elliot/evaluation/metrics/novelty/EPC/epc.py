"""
This is the implementation of the Expected Popularity Complement metric.
It proceeds from a user-wise computation, and average the values over the users.
"""


from elliot.evaluation.metrics.base_metric import BaseMetric
from elliot.utils.registry import metric_registry


@metric_registry.register()
class EPC(BaseMetric):
    r"""
    Expected Popularity Complement (EPC)

    This class represents the implementation of the Expected Popularity Complement recommendation metric.

    For further details, please refer to the `paper <https://dl.acm.org/doi/pdf/10.1145/2043932.2043955>`_

    Note:
         EPC can be read as the expected number of seen relevant recommended items not previously seen

    .. math::
       \mathrm{EPC}=C \sum_{i_{k} \in R} \operatorname{disc}(k) p\left(r e l \mid i_{k}, u\right)\left(1-p\left(\operatorname{seen} \mid t_{k}\right)\right)

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [EPC]
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

    def __user_EPC(self, user_recommendations, user, cutoff):
        """
        Per User Expected Popularity Complement
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """

        nov = 0
        norm = 0
        for r, (i, _) in enumerate(user_recommendations[:cutoff]):
            nov += self._relevance.get_rel(user, i) * self._relevance.logarithmic_ranking_discount(r) * self._item_novelty_dict.get(i, 1)
            norm += self._relevance.logarithmic_ranking_discount(r)

        if norm > 0:
            nov /= norm

        return nov

    # @staticmethod
    # def __discount_k(k):
    #     return (1 / math.log(k + 2)) * math.log(2)

    # def eval(self):
    #     """
    #     Evaluation function
    #     :return: the overall averaged value of Expected Popularity Complement
    #     """
    #
    #     item_count = {}
    #     for u_h in self._evaluation_objects.train_data.get_dict().values():
    #         for i in u_h.keys():
    #             item_count[i] = item_count.get(i, 0) + 1
    #
    #     num_users = len(self._evaluation_objects.train_data.get_dict())
    #     self._item_novelty_dict = {i: 1 - (v / num_users) for i, v in item_count.items()}
    #
    #     a = [self.__user_EPC(u_r, u, self._cutoff)
    #          for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))]
    #     return np.average(a)

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of Expected Popularity Complement per user
        """
        pop_cache = getattr(self._evaluation_objects, "pop_cache", None)
        if pop_cache and getattr(pop_cache, "item_novelty_epc", None) is not None:
            self._item_novelty_dict = pop_cache.item_novelty_epc
        else:
            item_count = {}
            for user_hist in self._evaluation_objects.train_data.get_dict().values():
                for item in user_hist.keys():
                    item_count[item] = item_count.get(item, 0) + 1
            num_users = len(self._evaluation_objects.train_data.get_dict())
            self._item_novelty_dict = {
                item: 1 - (count / num_users)
                for item, count in item_count.items()
            } if num_users > 0 else {}

        return {u: self.__user_EPC(u_r, u, self._cutoff)
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))}
