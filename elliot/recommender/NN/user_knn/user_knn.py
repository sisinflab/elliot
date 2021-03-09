"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pickle
import time

from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.NN.user_knn.user_knn_similarity import Similarity
from elliot.recommender.NN.user_knn.aiolli_ferrari import AiolliSimilarity
from elliot.recommender.base_recommender_model import init_charger

np.random.seed(42)


class UserKNN(RecMixin, BaseRecommenderModel):
    r"""
    GroupLens: An Open Architecture for Collaborative Filtering of Netnews

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/192844.192905>`_

    Args:
        neighbors: Number of item neighbors
        similarity: Similarity function
        implementation: Implementation type ('aiolli', 'classical')

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        UserKNN:
          meta:
            save_recs: True
          neighbors: 40
          similarity: cosine
          implementation: aiolli
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._random = np.random

        self._params_list = [
            ("_num_neighbors", "neighbors", "nn", 40, int, None),
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_implementation", "implementation", "imp", "standard", None, None)
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict
        if self._implementation == "aiolli":
            self._model = AiolliSimilarity(self._data, maxk=self._num_neighbors, shrink=100, similarity=self._similarity, normalize=True)
        else:
            self._model = Similarity(self._data, self._num_neighbors, self._similarity)

    def get_recommendations(self, k: int = 100):
        return {u: self._model.get_user_recs(u, k) for u in self._ratings.keys()}

    @property
    def name(self):
        return f"UserKNN_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        start = time.time()
        self._model.initialize()
        end = time.time()
        print(f"The similarity computation has taken: {end - start}")

        print(f"Transactions: {self._data.transactions}")

        best_metric_value = 0

        recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
        result_dict = self.evaluator.eval(recs)
        self._results.append(result_dict)
        print(f'Finished')

        if self._results[-1][self._validation_k]["val_results"][self._validation_metric] > best_metric_value:
            print("******************************************")
            if self._save_weights:
                with open(self._saving_filepath, "wb") as f:
                    pickle.dump(self._model.get_model_state(), f)
            if self._save_recs:
                store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")

    def restore_weights(self):
        try:
            with open(self._saving_filepath, "rb") as f:
                self._model.set_model_state(pickle.load(f))
            print(f"Model correctly Restored")

            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)
            self._results.append(result_dict)

            print("******************************************")
            if self._save_recs:
                store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")
            return True

        except Exception as ex:
            print(f"Error in model restoring operation! {ex}")

        return False
