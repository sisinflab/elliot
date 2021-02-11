"""
This is the implementation of the SRecall metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from elliot.evaluation.metrics.base_metric import BaseMetric


class SRecall(BaseMetric):
    """
    This class represents the implementation of the SRecall recommendation metric.
    Passing 'SRecall' to the metrics list will enable the computation of the metric.
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
        self._relevant_items = self._evaluation_objects.relevance.get_binary_relevance()
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

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of SRecall
        """
        return np.average(
            [SRecall.__user_srecall(u_r, self._cutoff, self._relevant_items[u], self._feature_map,self._total_features)
             for u, u_r in self._recommendations.items()]
        )

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall averaged value of SRecall
        """
        return {u: SRecall.__user_srecall(u_r, self._cutoff, self._relevant_items[u], self._feature_map, self._total_features)
             for u, u_r in self._recommendations.items()}

    @staticmethod
    def _load_attribute_file(attribute_file, separator='\t'):
        map = {}
        with open(attribute_file) as file:
            for line in file:
                line = line.split(separator)
                int_list = [int(i) for i in line[1:]]
                map[int(line[0])] = list(set(int_list))
        return map

