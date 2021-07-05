"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

from ast import literal_eval as make_tuple

import numpy as np
from tqdm import tqdm

from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.neural.NPR.neural_personalized_ranking_model import NPRModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation


class NPR(RecMixin, BaseRecommenderModel):
    r"""
    Neural Personalized Ranking for Image Recommendation
    (Model without visual features)

    For further details, please refer to the `paper <https://dl.acm.org/citation.cfm?id=3159728>`_

    Args:
        mf_factors: Number of MF latent factors
        mlp_hidden_size: List of units for each layer
        lr: Learning rate
        l_w: Regularization coefficient
        dropout: Dropout rate

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        NPR:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          mf_factors: 100
          mlp_hidden_size:  (64,32)
          lr: 0.001
          l_w: 0.001
          dropout: 0.45
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._sampler = cs.Sampler(self._data.i_train_dict)

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_l_w", "l_w", "l_w", 0.001, None, None),
            ("_mf_factors", "mf_factors", "mffactors", 100, None, None),
            ("_mlp_hidden_size", "mlp_hidden_size", "mlpunits", "(64,32)", lambda x: list(make_tuple(str(x))), lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_dropout", "dropout", "drop", 0.45, None, None)
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._model = NPRModel(self._num_users,
                               self._num_items,
                               self._mf_factors,
                               self._l_w,
                               self._mlp_hidden_size,
                               self._dropout,
                               self._learning_rate,
                               self._seed)

    @property
    def name(self):
        return "NPR"\
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
            predictions = self._model.get_recs(
                (
                    np.repeat(np.array(list(range(offset, offset_stop)))[:, None], repeats=self._num_items, axis=1),
                    np.array([self._i_items_set for _ in range(offset, offset_stop)])
                )
            )
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)

            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test
