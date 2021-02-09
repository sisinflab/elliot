"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Felice Antonio Merra'
__email__ = 'felice.merra@poliba.it'

import time
import numpy as np
import pickle
from ast import literal_eval as make_tuple
from tqdm import tqdm

from dataset.samplers import pointwise_pos_neg_sampler as pws
from evaluation.evaluator import Evaluator
from recommender.latent_factor_models.Slim.slim_model import SlimModel
from recommender.recommender_utils_mixin import RecMixin
from utils.folder import build_model_folder
from utils.write import store_recommendation

from recommender.base_recommender_model import BaseRecommenderModel

np.random.seed(42)


class Slim(RecMixin, BaseRecommenderModel):

    def __init__(self, data, config, params, *args, **kwargs):
        super().__init__(data, config, params, *args, **kwargs)

        self._restore = getattr(self._params, "restore", False)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._l1_ratio = self._params.l1_ratio
        self._alpha = self._params.alpha

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._model = SlimModel(self._data, self._num_users, self._num_items, self._l1_ratio, self._alpha, self._epochs)

        self._iteration = 0

        self.evaluator = Evaluator(self._data, self._params)

        self._params.name = self.name

        build_model_folder(self._config.path_output_rec_weight, self.name)
        self._saving_filepath = f'{self._config.path_output_rec_weight}{self.name}/best-weights-{self.name}'

    @property
    def name(self):
        return "Slim" \
               + "_e:" + str(self._epochs) \
               + "_l1_ratio:" + str(self._l1_ratio) \
               + "_alpha:" + str(self._alpha) \
               + f"_{self.get_params_shortcut()}"

    def get_recommendations(self, k: int = 100):
        return {u: self._model.get_user_recs(u, k) for u in self._ratings.keys()}

    def predict(self, u: int, i: int):
        """
        Get prediction on the user item pair.

        Returns:
            A single float vaue.
        """
        return self._model.predict(u, i)

    def train(self):

        if not self.restore_weights(self._epochs):
            self._model.train(self._verbose)

        recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
        result_dict = self.evaluator.eval(recs)
        self._results.append(result_dict)

        print("******************************************")
        if self._save_weights:
            with open(self._saving_filepath, "wb") as f:
                pickle.dump(self._model.get_model_state(), f)
        if self._save_recs:
            store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")

    def restore_weights(self, epochs):
        if self._restore_epochs == epochs:
            try:
                with open(self._saving_filepath, "rb") as f:
                    self._model.set_model_state(pickle.load(f))
                print(f"Model correctly Restored at Epoch: {self._restore_epochs}")
                return True
            except Exception as ex:
                print(f"Error in model restoring operation! {ex}")
        return False