"""
Created on April 4, 2020
Tensorflow 2.1.0 implementation of APR.
@author Anonymized
"""

import operator

import numpy as np

from elliot.evaluation.evaluator import Evaluator
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation
from elliot.recommender.base_recommender_model import init_charger

np.random.seed(0)


class MostPop(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a Most Popular recommender.
        :param data: data loader object
        :param path_output_rec_result: path to the directory rec. results
        :param path_output_rec_weight: path to the directory rec. model parameters
        :param args: parameters
        """
        self._random = np.random
        self.evaluator = Evaluator(self._data, self._params)

        self._pop_items = {self._data.private_items[p]: pop for p, pop in enumerate(self._data.sp_i_train.astype(bool).sum(axis=0).tolist()[0])}
        self._sorted_pop_items = dict(sorted(self._pop_items.items(), key=operator.itemgetter(1), reverse=True))

    @property
    def name(self):
        return "MostPop"

    def train(self):
        recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
        result_dict = self.evaluator.eval(recs)
        self._results.append(result_dict)

        if self._save_recs:
            store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")

    def get_recommendations(self, top_k):
        n_items = self._num_items
        sorted_pop_items = self._sorted_pop_items
        ratings = self._data.train_dict

        r = {}
        for u, i_s in ratings.items():
            l = []
            ui = set(i_s.keys())

            lui = len(ui)

            local_k = min(top_k, n_items - lui)

            for item, pop in sorted_pop_items.items():
                if item in ui:
                    continue
                else:
                    l.append((item, pop))
                if len(l) >= local_k:
                    break
            r[u] = l
        return r
