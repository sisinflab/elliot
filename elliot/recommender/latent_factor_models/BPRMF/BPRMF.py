"""
Module description:

"""
from tqdm import tqdm

from elliot.recommender.latent_factor_models.BPRMF.BPRMF_model import MFModel

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class BPRMF(RecMixin, BaseRecommenderModel):
    r"""
    Bayesian Personalized Ranking with Matrix Factorization

    For further details, please refer to the `paper <https://arxiv.org/abs/1205.2618.pdf>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        bias_regularization: Regularization coefficient for the bias
        user_regularization: Regularization coefficient for user latent factors
        positive_item_regularization: Regularization coefficient for positive item latent factors
        negative_item_regularization: Regularization coefficient for negative item latent factors
        update_negative_item_factors:
        update_users:
        update_items:
        update_bias:


    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        BPRMF:
          meta:
            save_recs: True
          epochs: 10
          factors: 10
          lr: 0.001
          bias_regularization: 0
          user_regularization: 0.0025
          positive_item_regularization: 0.0025
          negative_item_regularization: 0.0025
          update_negative_item_factors: True
          update_users: True
          update_items: True
          update_bias: True
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_factors", "factors", "f", 10, int, None),
            ("_learning_rate", "lr", "lr", 0.05, None, None),
            ("_bias_regularization", "bias_regularization", "bias_reg", 0, None, None),
            ("_user_regularization", "user_regularization", "u_reg", 0.0025,
             None, None),
            ("_positive_item_regularization", "positive_item_regularization", "pos_i_reg", 0.0025,
             None, None),
            ("_negative_item_regularization", "negative_item_regularization", "neg_i_reg", 0.00025,
             None, None),
            ("_update_negative_item_factors", "update_negative_item_factors", "up_neg_i_f", True,
             None, None),
            ("_update_users", "update_users", "up_u", True, None, None),
            ("_update_items", "update_items", "up_i", True, None, None),
            ("_update_bias", "update_bias", "up_b", True, None, None),
        ]
        self.autoset_params()

        self._batch_size = 1
        self._ratings = self._data.train_dict

        self._model = MFModel(self._factors,
                              self._data,
                              self._learning_rate,
                              self._user_regularization,
                              self._bias_regularization,
                              self._positive_item_regularization,
                              self._negative_item_regularization,
                              self._seed)
        self._sampler = cs.Sampler(self._data.i_train_dict)

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_predictions(u, mask, k) for u in self._ratings.keys()}

    @property
    def name(self):
        return "BPRMF" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        print(f"Transactions: {self._data.transactions}")

        for it in self.iterate(self._epochs):
            print(f"\n********** Iteration: {it + 1}")
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    self._model.train_step(batch)
                    t.update()

            self.evaluate(it)

