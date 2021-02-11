"""
This is the implementation of the User MAD ranking metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import math

import typing as t
import numpy as np
import pandas as pd
from elliot.evaluation.metrics.base_metric import BaseMetric


class UserMADranking(BaseMetric):
    """
    This class represents the implementation of the User MAD ranking recommendation metric.
    Passing 'UserMADranking' to the metrics list will enable the computation of the metric.

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
        self.rel_threshold = self._evaluation_objects.relevance._rel_threshold

        self._user_clustering = pd.read_csv(self._additional_data["clustering_file"], sep="\t", header=None)
        self._n_clusters = self._user_clustering[1].nunique()
        self._user_clustering = dict(zip(self._user_clustering[0], self._user_clustering[1]))
        self._sum = np.zeros(self._n_clusters)
        self._n_users = np.zeros(self._n_clusters)

    def name(self):
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return f"UserMADranking_{self._additional_data['clustering_name']}"

    @staticmethod
    def __user_mad(user_recommendations, relevance_map, cutoff):
        """
        Per User User MAD ranking
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        return UserMADranking.compute_user_ndcg(user_recommendations, relevance_map, cutoff)

    @staticmethod
    def compute_discount(k: int) -> float:
        """
        Method to compute logarithmic discount
        :param k:
        :return:
        """
        return 1 / math.log(k + 2) * math.log(2)

    @staticmethod
    def compute_idcg(gain_map: t.Dict, cutoff: int) -> float:
        """
        Method to compute Ideal Discounted Cumulative Gain
        :param gain_map:
        :param cutoff:
        :return:
        """
        gains: t.List = sorted(list(gain_map.values()))
        n: int = min(len(gains), cutoff)
        m: int = len(gains)
        return sum(map(lambda g, r: gains[m - r - 1] * UserMADranking.compute_discount(r), gains, range(n)))

    @staticmethod
    def compute_user_ndcg(user_recommendations: t.List, user_gain_map: t.Dict, cutoff: int) -> float:
        """
        Method to compute normalized Discounted Cumulative Gain
        :param sorted_item_predictions:
        :param gain_map:
        :param cutoff:
        :return:
        """
        idcg: float = UserMADranking.compute_idcg(user_gain_map, cutoff)
        dcg: float = sum(
            [user_gain_map.get(x, 0) * UserMADranking.compute_discount(r)
             for r, x in enumerate([item for item, _ in user_recommendations]) if r < cutoff])
        return dcg / idcg if dcg > 0 else 0

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of User MAD ranking
        """
        for u, u_r in self._recommendations.items():
            v = UserMADranking.__user_mad(u_r, self._relevance_map[u], self._cutoff)
            cluster = self._user_clustering.get(u, None)
            if cluster is not None:
                self._sum[cluster] += v
                self._n_users[cluster] += 1

        avg = [self._sum[i]/self._n_users[i] for i in range(self._n_clusters)]
        differences = []
        for i in range(self._n_clusters):
            for j in range(i+1,self._n_clusters):
                differences.append(abs(avg[i] - avg[j]))
        return np.average(differences)

    def get(self):
        return [self]

