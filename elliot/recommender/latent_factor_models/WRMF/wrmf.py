"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pickle

from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation
from elliot.recommender.latent_factor_models.WRMF.wrmf_model import WRMFModel
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger


class WRMF(RecMixin, BaseRecommenderModel):
    r"""
    Weighted XXX Matrix Factorization

    For further details, please refer to the `paper <https://archive.siam.org/meetings/sdm06/proceedings/059zhangs2.pdf>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        alpha:
        reg: Regularization coefficient

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        WRMF:
          meta:
            save_recs: True
          epochs: 10
          factors: 50
          alpha: 1
          reg: 0.1
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_factors", "factors", "factors", 10, None, None),
            ("_alpha", "alpha", "alpha", 1, None, None),
            ("_reg", "reg", "reg", 0.1, None, None)
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train

        self._model = WRMFModel(self._factors, self._data, self._nprandom, self._alpha, self._reg)

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
        return "WRMF" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            self._model.train_step()

            print("Iteration Finished")

            self.evaluate(it)

