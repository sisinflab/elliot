"""
This is the implementation of the SRecall metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from elliot.evaluation.metrics.base_metric import BaseMetric


class SRecall(BaseMetric):
    r"""
    Subtopic Recall

    This class represents the implementation of the Subtopic Recall (S-Recall) recommendation metric.

    For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/2795403.2795405>`_

    .. math::
        \mathrm {SRecall}=\frac{\left|\cup_{i=1}^{K} {subtopics}\left(d_{i}\right)\right|}{n_{A}}

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [SRecall]
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
        self._feature_map = SRecall._load_attribute_file(additional_data["feature_data"])
        self._total_features = len({topic for item in eval_objects.data.items for topic in self._feature_map.get(item, [])})

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "SRecall"

    @staticmethod
    def __user_srecall(user_recommendations, cutoff, user_relevant_items, feature_map, total_features):
        """
        Per User SRecall
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        subtopics = len({topic for i, _ in user_recommendations[:cutoff] if i in user_relevant_items for topic in feature_map.get(i, [])})
        return subtopics/total_features if total_features != 0 else 0

    # def eval(self):
    #     """
    #     Evaluation function
    #     :return: the overall averaged value of SRecall
    #     """
    #     return np.average(
    #         [SRecall.__user_srecall(u_r, self._cutoff, self._relevant_items[u], self._feature_map,self._total_features)
    #          for u, u_r in self._recommendations.items() if len(self._relevant_items[u])]
    #     )

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of SRecall
        """
        return {u: SRecall.__user_srecall(u_r, self._cutoff, self._relevance.get_user_rel(u), self._feature_map, self._total_features)
             for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u))}

    @staticmethod
    def _load_attribute_file(attribute_file, separator='\t'):
        map = {}
        with open(attribute_file) as file:
            for line in file:
                line = line.split(separator)
                int_list = [int(i) for i in line[1:]]
                map[int(line[0])] = list(set(int_list))
        return map

