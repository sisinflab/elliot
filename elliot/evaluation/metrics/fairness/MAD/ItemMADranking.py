"""
This is the implementation of the Item MAD ranking metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pandas as pd
from elliot.evaluation.metrics.base_metric import BaseMetric


class ItemMADranking(BaseMetric):
    """
    This class represents the implementation of the Item MAD ranking recommendation metric.
    Passing 'ItemMADranking' to the metrics list will enable the computation of the metric.

    Deldjoo, Yashar, Vito Walter Anelli, Hamed Zamani, Alejandro Bellogin, and Tommaso Di Noia.
    "A flexible framework for evaluating user and item fairness in recommender systems."
    User Modeling and User-Adapted Interaction (2020): 1-47.
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
        self._relevance_map = self._evaluation_objects.relevance.get_discounted_relevance()
        self._item_clustering = pd.read_csv(self._additional_data["clustering_file"], sep="\t", header=None)
        self._n_clusters = self._item_clustering[1].nunique()
        self._item_clustering = dict(zip(self._item_clustering[0], self._item_clustering[1]))
        self._sum = np.zeros(self._n_clusters)
        self._n_items = np.zeros(self._n_clusters)

        self._item_count = {}
        self._item_gain = {}

    def name(self):
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return f"ItemMADranking_{self._additional_data['clustering_name']}"

    def __item_mad(self, user_recommendations, cutoff, user_gain_map):
        """
        Per User Item MAD ranking
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        for i, r in user_recommendations[:cutoff]:
            self._item_count[i] = self._item_count.get(i, 0) + 1
            self._item_gain[i] = self._item_gain.get(i, 0) + user_gain_map.get(i, 0)

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of Item MAD ranking
        """

        for u, u_r in self._recommendations.items():
            self.__item_mad(u_r, self._cutoff, self._relevance_map[u])

        for item, gain in self._item_gain.items():
            v = gain/self._item_count[item]
            cluster = self._item_clustering.get(item, None)

            if cluster is not None:
                self._sum[cluster] += v
                self._n_items[cluster] += 1

        avg = [self._sum[i]/self._n_items[i] for i in range(self._n_clusters)]
        differences = []
        for i in range(self._n_clusters):
            for j in range(i+1, self._n_clusters):
                differences.append(abs(avg[i] - avg[j]))
        return np.average(differences)

    def get(self):
        return [self]

