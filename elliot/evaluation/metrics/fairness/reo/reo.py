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
    r"""
    This class represents the implementation of the Ranking-based Equal Opportunity (REO) recommendation metric.
    Passing 'REO' to the metrics list will enable the computation of the metric.


    .. _PopREO:   Measuring and Mitigating Item Under-Recommendation Bias in Personalized Ranking Systems."
    Zhu, Ziwei, Jianling Wang, and James Caverlee.
    Proceedings of the 43rd International ACM SIGIR
    Conference on Research and Development in Information Retrieval. 2020.

    .. math::
        \mathrm {REO}=\frac{{std}\left(P\left(R @ k \mid g=g_{1}, y=1\right) \ldots P\left(R(a) k=g_{A}, y=1\right)\right)}
        {{mean}\left(P\left(R @ k \mid g=g_{1}, y=1\right) \ldots P\left(R @ k \mid g=g_{A}, y=1\right)\right)}

    :math:`P\left(R @ k \mid g=g_{a}, y=1\right)` is `\frac{\sum_{u=1}^{N} \sum_{i=1}^{k} G_{g_{a}}\left(R_{u, i}\right) Y\left(u, R_{u, i}\right)}
    {\sum_{u=1}^{N} \sum_{i \in I \backslash I_{u}^{+}} G_{g_{a}}(i) Y(u, i)}`

    :math:`Y\left(u, R_{u, i}\right)` identifies the ground-truth label of a user-item pair `\left(u, R_{u, i}\right)`,
    if item `R_{u, i}` is liked by user 𝑢, returns 1, otherwise 0

    :math:`\sum_{i=1}^{k} G_{g_{a}}\left(R_{u, i}\right) Y\left(u, R_{u, i}\right)`
    counts how many items in test set from group `{g_a}` are ranked in top-𝑘 for user u

    :math:`\sum_{i \in I \backslash I_{u}^{+}} G_{g_{a}}(i) Y(u, i)`
    counts the total number of items from group `{g_a}` 𝑎 in test set for user u

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
            if len(self._relevance.get_user_rel(u)):
                self.__user_pop_reo(u_r, set(self._train[u].keys()), self._cutoff, set(self._relevance.get_user_rel(u)))

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

