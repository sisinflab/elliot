"""
Module description:

"""


__version__ = '0.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle

import numpy as np

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.latent_factor_models.Slim.slim_model import SlimModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

np.random.seed(42)


class Slim(RecMixin, BaseRecommenderModel):
    r"""
    Sparse Linear Methods

    For further details, please refer to the `paper <http://glaros.dtc.umn.edu/gkhome/node/774>`_

    Args:
        l1_ratio:
        alpha:

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        Slim:
          meta:
            save_recs: True
          l1_ratio: 0.001
          alpha: 0.001
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_l1_ratio", "l1_ratio", "l1", 0.001, None, None),
            ("_alpha", "alpha", "alpha", 0.001, None, None),
        ]

        self.autoset_params()

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._model = SlimModel(self._data, self._num_users, self._num_items, self._l1_ratio, self._alpha, self._epochs)

    @property
    def name(self):
        return "Slim" \
               + "_e:" + str(self._epochs) \
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
        if self._restore:
            return self.restore_weights()

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