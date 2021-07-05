"""
Module description:
Lemire, Daniel, and Anna Maclachlan. "Slope one predictors for online rating-based collaborative filtering."
Proceedings of the 2005 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics
"""


__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle

import numpy as np

from elliot.recommender.algebric.slope_one.slope_one_model import SlopeOneModel
from elliot.recommender.base_recommender_model import BaseRecommenderModel, init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation


class SlopeOne(RecMixin, BaseRecommenderModel):
    r"""
    Slope One Predictors for Online Rating-Based Collaborative Filtering

    For further details, please refer to the `paper <https://arxiv.org/abs/cs/0702144>`_

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        SlopeOne:
          meta:
            save_recs: True
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        super().__init__(data, config, params, *args, **kwargs)

        self._restore = getattr(self._params, "restore", False)
        self._num_items = self._data.num_items
        self._num_users = self._data.num_users

        self._ratings = self._data.train_dict
        self._i_ratings = self._data.i_train_dict

        self._model = SlopeOneModel(self._data)

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_recs(u, mask, k) for u in self._data.train_dict.keys()}

    @property
    def name(self):
        return f"SlopeOne"

    def train(self):
        if self._restore:
            return self.restore_weights()

        self._model.initialize()

        self.evaluate()

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
