"""
Module description:

"""
from recommender.latent_factor_models.PureSVD.pure_svd_model import PureSVDModel
from recommender.latent_factor_models.WRMF.wrmf_model import WRMFModel

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import time
import numpy as np
import pickle
from tqdm import tqdm

from dataset.samplers import pairwise_sampler as ps
from evaluation.evaluator import Evaluator
from recommender.recommender_utils_mixin import RecMixin
from utils.folder import build_model_folder
from utils.write import store_recommendation

from recommender.base_recommender_model import BaseRecommenderModel

np.random.seed(42)


class PureSVD(RecMixin, BaseRecommenderModel):

    def __init__(self, data, config, params, *args, **kwargs):
        super().__init__(data, config, params, *args, **kwargs)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random

        self._factors = self._params.embed_k

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._model = PureSVDModel(self._factors, self._sp_i_train, 42)

        self._iteration = 0

        self.evaluator = Evaluator(self._data, self._params)

        self._params.name = self.name

        build_model_folder(self._config.path_output_rec_weight, self.name)
        self._saving_filepath = f'{self._config.path_output_rec_weight}{self.name}best-weights-{self.name}'

    def get_recommendations(self, k: int = 100):
        return {u: self._model.get_user_recs(u, k) for u in self._ratings.keys()}

    def predict(self, u: int, i: int):
        """
        Get prediction on the user item pair.

        Returns:
            A single float vaue.
        """
        return self._model.predict(u, i)

    @property
    def name(self):
        return "PureSVD" + "-factors:" + str(self._params.embed_k)

    def train(self):

        best_metric_value = 0
        self._update_count = 0

        for it in range(self._epochs):
            self.restore_weights(it)
            loss = 0
            steps = 0
            self._model.train_step()

            if not (it + 1) % self._validation_rate:
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                result_dict = self.evaluator.eval(recs)
                self._results.append(result_dict)

                print(f'Epoch {(it + 1)}/{self._epochs} loss {loss/steps:.5f}')

                if self._results[-1][self._validation_k]["val_results"][self._validation_metric] > best_metric_value:
                    print("******************************************")
                    best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                    if self._save_weights:
                        with open(self._saving_filepath, "wb") as f:
                            pickle.dump(self._model.get_model_state(), f)
                    if self._save_recs:
                        store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

    def restore_weights(self, it):
        if self._restore_epochs == it:
            try:
                with open(self._saving_filepath, "rb") as f:
                    self._model.set_model_state(pickle.load(f))
                print(f"Model correctly Restored at Epoch: {self._restore_epochs}")
                return True
            except Exception as ex:
                print(f"Error in model restoring operation! {ex}")
        return False
