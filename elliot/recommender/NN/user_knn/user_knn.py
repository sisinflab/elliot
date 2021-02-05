"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pickle
import time

from evaluation.evaluator import Evaluator
from recommender.recommender_utils_mixin import RecMixin
from utils.folder import build_model_folder
from utils.write import store_recommendation

from recommender.base_recommender_model import BaseRecommenderModel
from recommender.NN.user_knn.user_knn_similarity import Similarity
from recommender.NN.user_knn.aiolli_ferrari import AiolliSimilarity

np.random.seed(42)


class UserKNN(RecMixin, BaseRecommenderModel):

    def __init__(self, data, config, params, *args, **kwargs):
        super().__init__(data, config, params, *args, **kwargs)

        self._restore = getattr(self._params, "restore", False)
        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random

        self._params_list = [
            ("_num_neighbors", "neighbors", "nn", 40, None, None),
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_implementation", "implementation", "imp", "standard", None, None)
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict
        if self._implementation == "aiolli":
            self._datamodel = AiolliSimilarity(self._data, maxk=self._num_neighbors, shrink=100, similarity=self._similarity, normalize=True)
        else:
            self._datamodel = Similarity(self._data, self._num_neighbors, self._similarity)

        self._params.name = self.name

        build_model_folder(self._config.path_output_rec_weight, self.name)
        self._saving_filepath = f'{self._config.path_output_rec_weight}{self.name}/best-weights-{self.name}'

        start = time.time()
        if self._restore:
            self.restore_weights()
        else:
            self._datamodel.initialize()
        end = time.time()
        print(f"The similarity computation has taken: {end - start}")

        self.evaluator = Evaluator(self._data, self._params)

        if self._save_weights:
            with open(self._saving_filepath, "wb") as f:
                print("Saving Model")
                pickle.dump(self._datamodel.get_model_state(), f)

    def get_recommendations(self, k: int = 100):
        return {u: self._datamodel.get_user_recs(u, k) for u in self._ratings.keys()}

    @property
    def name(self):
        return f"UserKNN_{self.get_params_shortcut()}"

    def train(self):

        print(f"Transactions: {self._data.transactions}")
        best_metric_value = 0

        recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
        result_dict = self.evaluator.eval(recs)
        self._results.append(result_dict)
        print(f'Finished')

        if self._results[-1][self._validation_k]["val_results"][self._validation_metric] > best_metric_value:
            print("******************************************")
            if self._save_recs:
                store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")


    def restore_weights(self):
        try:
            with open(self._saving_filepath, "rb") as f:
                self._datamodel.set_model_state(pickle.load(f))
            print(f"Model correctly Restored")
            return True
        except Exception as ex:
            print(f"Error in model restoring operation! {ex}")