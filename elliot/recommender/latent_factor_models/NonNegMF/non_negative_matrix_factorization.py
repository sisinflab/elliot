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
from recommender.latent_factor_models.NonNegMF.non_negative_matrix_factorization_model import NonNegMFModel
from recommender.recommender_utils_mixin import RecMixin
from utils.folder import build_model_folder
from utils.write import store_recommendation

from recommender.base_recommender_model import BaseRecommenderModel

np.random.seed(42)


class NonNegMF(RecMixin, BaseRecommenderModel):

    def __init__(self, data, config, params, *args, **kwargs):
        super().__init__(data, config, params, *args, **kwargs)

        self._restore = getattr(self._params, "restore", False)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random
        self._sample_negative_items_empirically = True

        self._params_list = [
            ("_factors", "factors", "factors", 10, None, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_l_w", "reg", "reg", 0.1, None, None),
        ]
        self.autoset_params()

        # self._learning_rate = self._params.lr
        # self._factors = self._params.factors
        # self._l_w = self._params.reg

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict
        self._global_mean = np.mean([r for user_items in self._data.train_dict.values() for r in user_items.values()])
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._model = NonNegMFModel(self._data, self._num_users, self._num_items, self._global_mean, self._factors,
                                    self._l_w, self._learning_rate, random_seed=42)

        self._iteration = 0

        self.evaluator = Evaluator(self._data, self._params)

        self._params.name = self.name

        build_model_folder(self._config.path_output_rec_weight, self.name)
        self._saving_filepath = f'{self._config.path_output_rec_weight}{self.name}/best-weights-{self.name}'

    @property
    def name(self):
        return "NonNegMF" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
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
        print(f"Transactions: {self._data.transactions}")
        best_metric_value = -np.inf
        for it in range(self._epochs):
            self.restore_weights(it)
            print(f"\n********** Iteration: {it + 1}")
            self._iteration = it

            self._model.train_step()

            if not (it + 1) % self._validation_rate:
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                result_dict = self.evaluator.eval(recs)
                self._results.append(result_dict)

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
