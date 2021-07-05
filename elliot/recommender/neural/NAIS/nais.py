"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'
__paper__ = 'FISM: Factored Item Similarity Models for Top-N Recommender Systems by Santosh Kabbur, Xia Ning, and George Karypis'


import numpy as np
from tqdm import tqdm

from elliot.dataset.samplers import pointwise_pos_neg_ratio_ratings_sampler as pws
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.neural.NAIS.nais_model import NAIS_model
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation


class NAIS(RecMixin, BaseRecommenderModel):
    r"""
    NAIS: Neural Attentive Item Similarity Model for Recommendation

    For further details, please refer to the `paper <https://arxiv.org/abs/1809.07053>`_

    Args:
        factors: Number of latent factors
        algorithm: Type of user-item factor operation ('product', 'concat')
        weight_size: List of units for each layer
        lr: Learning rate
        l_w: Regularization coefficient
        l_b: Bias regularization coefficient
        alpha: Attention factor
        beta: Smoothing exponent
        neg_ratio: Ratio of negative sampled items, e.g., 0 = no items, 1 = all un-rated items

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        NAIS:
          meta:
            save_recs: True
          factors: 100
          batch_size: 512
          algorithm: concat
          weight_size: 32
          lr: 0.001
          l_w: 0.001
          l_b: 0.001
          alpha: 0.5
          beta: 0.5
          neg_ratio: 0.5
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """

        Create a NAIS instance.
        (see https://arxiv.org/pdf/1809.07053.pdf for details about the algorithm design choices).

        """

        self._params_list = [
            ("_factors", "factors", "factors", 100, None, None),
            ("_algorithm", "algorithm", "algorithm", "concat", None, None),
            ("_weight_size", "weight_size", "weight_size", 32, None, None),
            ("_lr", "lr", "lr", 0.001, None, None),
            ("_l_w", "l_w", "l_w", 0.001, None, None),
            ("_l_b", "l_b", "l_b", 0.001, None, None),
            ("_alpha", "alpha", "alpha", 0.5, lambda x: min(max(0, x), 1), None),
            ("_beta", "beta", "beta", 0.5, None, None),
            ("_neg_ratio", "neg_ratio", "neg_ratio", 0.5, None, None)
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._sampler = pws.Sampler(self._data.i_train_dict, self._data.sp_i_train_ratings, self._neg_ratio)

        self._model = NAIS_model(self._data,
                                 self._algorithm,
                                 self._weight_size,
                                 self._factors,
                                 self._lr,
                                 self._l_w,
                                 self._l_b,
                                 self._alpha,
                                 self._beta,
                                 self._num_users,
                                 self._num_items,
                                 self._seed)

    @property
    def name(self):
        return "NAIS" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss.numpy()/(it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.batch_predict(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

