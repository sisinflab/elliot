"""
This is the implementation of the User MAD ranking metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import math

import typing as t
import numpy as np
import pandas as pd
from elliot.evaluation.metrics.base_metric import BaseMetric


class UserMADranking(BaseMetric):
    r"""
    User MAD Ranking-based

    This class represents the implementation of the User MAD ranking recommendation metric.

    For further details, please refer to the `paper <https://link.springer.com/article/10.1007/s11257-020-09285-1>`_

     .. math::
        \mathrm {MAD}={avg}_{i, j}({MAD}(R^{(i)}, R^{(j)}))

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        complex_metrics:
        - metric: UserMADranking
          clustering_name: Happiness
          clustering_file: ../data/movielens_1m/u_happy.tsv
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
        self._relevance = self._evaluation_objects.relevance.discounted_relevance
        # self.rel_threshold = self._evaluation_objects.relevance._rel_threshold

        self._user_clustering_path = self._additional_data.get("clustering_file", False)
        self._user_clustering_name = self._additional_data.get("clustering_name", "")
        if self._user_clustering_path:
            self._user_clustering = pd.read_csv(self._additional_data["clustering_file"], sep="\t", header=None)
            self._n_clusters = self._user_clustering[1].nunique()
            self._user_clustering = dict(zip(self._user_clustering[0], self._user_clustering[1]))
        else:
            self._n_clusters = 1
            self._user_clustering = {}

        self._sum = np.zeros(self._n_clusters)
        self._n_users = np.zeros(self._n_clusters)

    def name(self):
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return f"UserMADranking_{self._user_clustering_name}"

    def __user_mad(self, user_recommendations, user, cutoff):
        """
        Per User User MAD ranking
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        return self.compute_user_ndcg(user_recommendations, user, cutoff)

    # @staticmethod
    # def compute_discount(k: int) -> float:
    #     """
    #     Method to compute logarithmic discount
    #     :param k:
    #     :return:
    #     """
    #     return 1 / math.log(k + 2) * math.log(2)

    def compute_idcg(self, user: int, cutoff: int) -> float:
        """
        Method to compute Ideal Discounted Cumulative Gain
        :param gain_map:
        :param cutoff:
        :return:
        """
        gains: t.List = sorted(list(self._relevance.get_user_rel_gains(user).values()))
        n: int = min(len(gains), cutoff)
        m: int = len(gains)
        return sum(map(lambda g, r: gains[m - r - 1] * self._relevance.logarithmic_ranking_discount(r), gains, range(n)))

    def compute_user_ndcg(self, user_recommendations: t.List, user: int, cutoff: int) -> float:
        """
        Method to compute normalized Discounted Cumulative Gain
        :param sorted_item_predictions:
        :param gain_map:
        :param cutoff:
        :return:
        """
        idcg: float = self.compute_idcg(user, cutoff)
        dcg: float = sum(
            [self._relevance.get_rel(user, x) * self._relevance.logarithmic_ranking_discount(r)
             for r, x in enumerate([item for item, _ in user_recommendations]) if r < cutoff])
        return dcg / idcg if dcg > 0 else 0

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of User MAD ranking
        """
        for u, u_r in self._recommendations.items():
            if len(self._relevance.get_user_rel(u)):
                v = self.__user_mad(u_r, u, self._cutoff)
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

