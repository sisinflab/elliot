"""
Created on April 4, 2020
Tensorflow 2.1.0 implementation of APR.
@author Anonymized
"""

import numpy as np

from evaluation.evaluator import Evaluator
from recommender.base_recommender_model import BaseRecommenderModel
from recommender.recommender_utils_mixin import RecMixin
from utils import logging
from utils.folder import build_model_folder
from utils.write import store_recommendation



class Random(RecMixin, BaseRecommenderModel):

    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a Random recommender.
        :param data: data loader object
        :param path_output_rec_result: path to the directory rec. results
        :param path_output_rec_weight: path to the directory rec. model parameters
        :param args: parameters
        """
        super().__init__(data, config, params, *args, **kwargs)
        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random
        self.evaluator = Evaluator(self._data, self._params)

        self._params_list = [
            ("_seed", "random_seed", "seed", 42, None, None)
        ]
        self.autoset_params()

        np.random.seed(self._seed)

        self._params.name = self.name

        build_model_folder(self._config.path_output_rec_weight, self.name)
        self._saving_filepath = f'{self._config.path_output_rec_weight}{self.name}/best-weights-{self.name}'
        self.logger = logging.get_logger(self.__class__.__name__)

    @property
    def name(self):
        return f"Random_{self.get_params_shortcut()}"

    def train(self):
        recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
        result_dict = self.evaluator.eval(recs)
        self._results.append(result_dict)

        if self._save_recs:
            store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")

    def get_recommendations(self, top_k):
        r_int = np.random.randint
        n_items = self._num_items
        items = self._data.items
        ratings = self._data.train_dict

        r = {}
        for u, i_s in ratings.items():
            l = []
            ui = set(i_s.keys())
            lui = len(ui)
            if lui+top_k >= n_items:
                r[u] = l
                continue
            for index in range(top_k):
                j = items[r_int(n_items)]
                while j in ui:
                    j = items[r_int(n_items)]
                l.append((j, 1))
            r[u] = l
        return r
