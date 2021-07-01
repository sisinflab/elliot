"""
Module description:

"""


__version__ = '0.3'
__author__ = 'Massimo Quadrana, Vito Walter Anelli, Claudio Pomo, Felice Antonio Merra'
__email__ = 'mquadrana@pandora.com, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, felice.merra@poliba.it'

import pickle

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.latent_factor_models.Slim.slim_model import SlimModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation


class Slim(RecMixin, BaseRecommenderModel):
    r"""
    Train a Sparse Linear Methods (SLIM) item similarity model.
        NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
              make use of half the cores available
        See:
            Efficient Top-N Recommendation by Linear Regression,
            M. Levy and K. Jack, LSRS workshop at RecSys 2013.

            SLIM: Sparse linear methods for top-n recommender systems,
            X. Ning and G. Karypis, ICDM 2011.
            For further details, please refer to the `paper <http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf>`_



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
            ("_l1_ratio", "l1_ratio", "l1", 0.001, float, None),
            ("_alpha", "alpha", "alpha", 0.001, float, None),
            ("_neighborhood", "neighborhood", "neighborhood", 10, int, None)
        ]

        self.autoset_params()

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._model = SlimModel(self._data, self._num_users, self._num_items, self._l1_ratio, self._alpha,
                                self._epochs, self._neighborhood, self._seed)

    @property
    def name(self):
        return "Slim" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_recs(u, mask, k) for u in self._data.train_dict.keys()}

    def get_recommendations(self, k: int = 10):
        self._model.prepare_predictions()

        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test


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

        self.evaluate()

