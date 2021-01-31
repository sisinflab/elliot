"""
This is the implementation of the User MAD rating metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pandas as pd
from evaluation.metrics.base_metric import BaseMetric


class UserMADrating(BaseMetric):
    """
    This class represents the implementation of the User MAD rating recommendation metric.
    Passing 'UserMADrating' to the metrics list will enable the computation of the metric.

    Zhu, Ziwei, Xia Hu, and James Caverlee. "Fairness-aware tensor-based recommendation." Proceedings of the 27th ACM International Conference on Information and Knowledge Management. 2018.
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
        return f"UserMADrating_{self._additional_data['clustering_name']}"

    @staticmethod
    def __user_mad(user_recommendations, cutoff, user_relevant_items):
        """
        Per User User MAD rating
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Precision metric for the specific user
        """
        # return np.average([i[1] for i in user_recommendations if i[0] in user_relevant_items])
        return np.average([i[1] for i in user_recommendations[:cutoff]])

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of User MAD rating
        """
        for u, u_r in self._recommendations.items():
            v = UserMADrating.__user_mad(u_r, self._cutoff, self._relevant_items[u])
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

