"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pickle

from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

from elliot.recommender.latent_factor_models.PureSVD.pure_svd_model import PureSVDModel
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger

np.random.seed(42)


class PureSVD(RecMixin, BaseRecommenderModel):
    r"""
    PureSVD

    For further details, please refer to the `paper <https://link.springer.com/chapter/10.1007/978-0-387-85820-3_5>`_

    Args:
        factors: Number of latent factors
        seed: Random seed

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        PureSVD:
          meta:
            save_recs: True
          factors: 10
          seed: 42
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._random = np.random

        self._params_list = [
            ("_factors", "factors", "factors", 10, None, None),
            ("_seed", "seed", "seed", 42, None, None)
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._model = PureSVDModel(self._factors, self._data, self._seed)

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
        return f"PureSVD_{self.get_params_shortcut()}"

    def train(self):

        if self._restore:
            return self.restore_weights()

        self._model.train_step()

        print("Computation finished, producing recommendations")

        recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
        result_dict = self.evaluator.eval(recs)
        self._results.append(result_dict)

        print("******************************************")
        print("Saving to files")

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