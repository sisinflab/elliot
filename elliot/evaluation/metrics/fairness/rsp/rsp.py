"""
This is the implementation of the Ranking-based Statistical Parity (RSP) metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pandas as pd

from collections import Counter

from evaluation.metrics.base_metric import BaseMetric
from evaluation.metrics.metrics_utils import ProxyMetric

class RSP(BaseMetric):
    """
    This class represents the implementation of the Ranking-based Statistical Parity (RSP) recommendation metric.
    Passing 'RSP' to the metrics list will enable the computation of the metric.
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
        self._train = self._evaluation_objects.data.train_dict

        self._item_clustering_path = self._additional_data.get("clustering_file", False)

        if self._item_clustering_path:
            self._item_clustering = pd.read_csv(self._item_clustering_path, sep="\t", header=None, names=["id","cluster"])
            self._item_n_clusters = self._item_clustering['cluster'].nunique()
            self._item_clustering = self._item_clustering.groupby('cluster')['id'].apply(set).to_dict()
            self._item_clustering_name = self._additional_data['clustering_name']
        else:
            self._item_n_clusters = 1
            self._item_clustering = {}
            self._item_clustering_name = ""

        self._num = np.zeros(self._item_n_clusters)
        self._den = np.zeros(self._item_n_clusters)

        self.process()

    def name(self):
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return f"RSP_items:{self._item_clustering_name}"

    def __user_pop_rsp(self, user_recommendations, user_train, cutoff):
        """
        Per User Ranking-based Statistical Parity (RSP)
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Bias Disparity - Bias Recommendations metric for the specific user
        """
        recommended_items = set([i for i, _ in user_recommendations[:cutoff]])

        for i, i_set in self._item_clustering.items():
            self._num[i] += len(recommended_items & i_set)
            self._den[i] += len(i_set - user_train)

    def eval(self):
        pass

    def process(self):
        """
        Evaluation function
        :return: the overall value of Ranking-based Statistical Parity (RSP)
        """

        for u, u_r in self._recommendations.items():
            self.__user_pop_rsp(u_r, set(self._train[u].keys()), self._cutoff)

        PR = self._num / self._den

        self._metric_objs_list = []
        for i_category in range(self._item_n_clusters):
            self._metric_objs_list.append(ProxyMetric(name= f"RSP-ProbToBeRanked_items:{self._item_clustering_name}-{i_category}",
                                                      val=PR[i_category],
                                                      needs_full_recommendations=False))
        # Overall RSP
        self._metric_objs_list.append(ProxyMetric(name=f"RSP_items:{self._item_clustering_name}",
                                                  val=np.std(PR) / np.mean(PR),
                                                  needs_full_recommendations=False))

    def get(self):
        return self._metric_objs_list

