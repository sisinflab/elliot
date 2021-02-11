"""
This is the implementation of the Ranking-based Equal Opportunity (REO) metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pandas as pd

from elliot.evaluation.metrics.base_metric import BaseMetric
from elliot.evaluation.metrics.metrics_utils import ProxyMetric


class REO(BaseMetric):
    """
    This class represents the implementation of the Ranking-based Equal Opportunity (REO) recommendation metric.
    Passing 'REO' to the metrics list will enable the computation of the metric.


    Zhu, Ziwei, Jianling Wang, and James Caverlee. "Measuring and Mitigating Item Under-Recommendation Bias in Personalized Ranking Systems." Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2020.

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
        return f"REO_items:{self._item_clustering_name}"

    def __user_pop_reo(self, user_recommendations, user_train, cutoff, user_relevant_items):
        """
        Per User Ranking-based Equal Opportunity (REO)
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Ranking-based Equal Opportunity (REO) metric for the specific user
        """
        recommended_items = set([i for i, _ in user_recommendations[:cutoff] if i in user_relevant_items])

        for i, i_set in self._item_clustering.items():
            self._num[i] += len(recommended_items & i_set)
            self._den[i] += len((i_set  & user_relevant_items) - user_train)

    def eval(self):
        pass

    def process(self):
        """
        Evaluation function
        :return: the overall value of Ranking-based Equal Opportunity (REO)
        """

        for u, u_r in self._recommendations.items():
            if len(self._relevant_items[u]):
                self.__user_pop_reo(u_r, set(self._train[u].keys()), self._cutoff, set(self._relevant_items[u]))

        PR = self._num / self._den

        self._metric_objs_list = []
        for i_category in range(self._item_n_clusters):
            self._metric_objs_list.append(ProxyMetric(name= f"REO-ProbToBeRanked_items:{self._item_clustering_name}-{i_category}",
                                                      val=PR[i_category],
                                                      needs_full_recommendations=False))
        # Overall REO
        self._metric_objs_list.append(ProxyMetric(name=f"REO_items:{self._item_clustering_name}",
                                                  val=np.std(PR) / np.mean(PR),
                                                  needs_full_recommendations=False))

    def get(self):
        return self._metric_objs_list

