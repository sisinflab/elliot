"""
This is the implementation of the User MAD rating metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pandas as pd
from elliot.evaluation.metrics.base_metric import BaseMetric


class UserMADrating(BaseMetric):
    r"""
    User MAD Rating-based

    This class represents the implementation of the User MAD rating recommendation metric.

    For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/3269206.3271795>`_

    .. math::
        \mathrm {MAD}={avg}_{i, j}({MAD}(R^{(i)}, R^{(j)}))

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        complex_metrics:
        - metric: UserMADrating
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
        self._relevance = self._evaluation_objects.relevance.binary_relevance

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
        return f"UserMADrating_{self._user_clustering_name}"

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
            if len(self._relevance.get_user_rel(u)):
                v = UserMADrating.__user_mad(u_r, self._cutoff, self._relevance.get_user_rel(u))
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

