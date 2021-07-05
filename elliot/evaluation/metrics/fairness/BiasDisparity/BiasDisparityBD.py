"""
This is the implementation of the Bias Disparity metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pandas as pd

from collections import Counter

from . import BiasDisparityBR, BiasDisparityBS

from elliot.evaluation.metrics.base_metric import BaseMetric
from elliot.evaluation.metrics.metrics_utils import ProxyMetric


class BiasDisparityBD(BaseMetric):
    r"""
    Bias Disparity - Standard

    This class represents the implementation of the Bias Disparity recommendation metric.

    For further details, please refer to the `paper <https://arxiv.org/pdf/1811.01461>`_

    .. math::
        \mathrm {BD(G, C)}=\frac{B_{R}(G, C)-B_{S}(G, C)}{B_{S}(G, C)}

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        complex_metrics:
            - metric: BiasDisparityBD
              user_clustering_name: Happiness
              user_clustering_file: ../data/movielens_1m/u_happy.tsv
              item_clustering_name: ItemPopularity
              item_clustering_file: ../data/movielens_1m/i_pop.tsv
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

        self._item_clustering_path = self._additional_data.get("item_clustering_file", False)

        if self._item_clustering_path:
            self._item_clustering = pd.read_csv(self._item_clustering_path, sep="\t", header=None)
            self._item_n_clusters = self._item_clustering[1].nunique()
            self._item_clustering = dict(zip(self._item_clustering[0], self._item_clustering[1]))
            self._item_clustering_name = self._additional_data['item_clustering_name']
        else:
            self._item_n_clusters = 1
            self._item_clustering = {}
            self._item_clustering_name = ""

        self._user_clustering_path = self._additional_data.get("user_clustering_file", False)

        if self._user_clustering_path:
            self._user_clustering = pd.read_csv(self._user_clustering_path, sep="\t", header=None)
            self._user_n_clusters = self._user_clustering[1].nunique()
            self._user_clustering = dict(zip(self._user_clustering[0], self._user_clustering[1]))
            self._user_clustering_name = self._additional_data['user_clustering_name']
        else:
            self._user_n_clusters = 1
            self._user_clustering = {}
            self._user_clustering_name = ""

        self._category_sum = np.zeros((self._user_n_clusters,self._item_n_clusters))
        self._total_sum = np.zeros(self._user_n_clusters)

        self.process()

    def name(self):
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return f"BiasDisparityBD_users:{self._user_clustering_name}_items:{self._item_clustering_name}"

    def eval(self):
        pass

    def process(self):
        """
        Evaluation function
        :return: the overall value of Bias Disparity
        """

        BR = BiasDisparityBR(self._recommendations, self._config, self._params, self._evaluation_objects, self._additional_data).get_BR()
        BS = BiasDisparityBS(self._recommendations, self._config, self._params, self._evaluation_objects, self._additional_data).get_BS()

        BD = (BR - BS) / BS

        self._metric_objs_list = []
        for u_group in range(self._user_n_clusters):
            for i_category in range(self._item_n_clusters):
                self._metric_objs_list.append(ProxyMetric(name= f"BiasDisparityBD_users:{self._user_clustering_name}-{u_group}_items:{self._item_clustering_name}-{i_category}",
                                                          val=BD[u_group, i_category],
                                                          needs_full_recommendations=False))

    def get(self):
        return self._metric_objs_list

