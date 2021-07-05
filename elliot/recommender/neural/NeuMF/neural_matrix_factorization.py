"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import time

import numpy as np
from tqdm import tqdm

from elliot.recommender.neural.NeuMF import custom_sampler as cs
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.neural.NeuMF.neural_matrix_factorization_model import NeuralMatrixFactorizationModel
from elliot.recommender.recommender_utils_mixin import RecMixin


class NeuMF(RecMixin, BaseRecommenderModel):
    r"""
    Neural Collaborative Filtering

    For further details, please refer to the `paper <https://arxiv.org/abs/1708.05031>`_

    Args:
        mf_factors: Number of MF latent factors
        mlp_factors: Number of MLP latent factors
        mlp_hidden_size: List of units for each layer
        lr: Learning rate
        dropout: Dropout rate
        is_mf_train: Whether to train the MF embeddings
        is_mlp_train: Whether to train the MLP layers

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        NeuMF:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          mf_factors: 10
          mlp_factors: 10
          mlp_hidden_size: (64,32)
          lr: 0.001
          dropout: 0.0
          is_mf_train: True
          is_mlp_train: True
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_mf_factors", "mf_factors", "mffactors", 10, int, None),
            # If the user prefer a generalized model (WARNING: not coherent with the paper) can uncomment the following options
            #("_mlp_factors", "mlp_factors", "mlpfactors", 10, int, None),
            #("_mlp_hidden_size", "mlp_hidden_size", "mlpunits", "(64,32)", lambda x: list(make_tuple(str(x))), lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_dropout", "dropout", "drop", 0, None, None),
            ("_is_mf_train", "is_mf_train", "mftrain", True, None, None),
            ("_is_mlp_train", "is_mlp_train", "mlptrain", True, None, None),
            ("_m", "m", "m", 0, int, None)
        ]
        self.autoset_params()

        self._mlp_hidden_size = (self._mf_factors*4, self._mf_factors*2, self._mf_factors)
        self._mlp_factors = self._mf_factors

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._sampler = cs.Sampler(self._data.i_train_dict, self._m)

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._model = NeuralMatrixFactorizationModel(self._num_users, self._num_items, self._mf_factors,
                                                     self._mlp_factors, self._mlp_hidden_size,
                                                     self._dropout, self._is_mf_train, self._is_mlp_train,
                                                     self._learning_rate, self._seed)

    @property
    def name(self):
        return "NeuMF"\
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions * (self._m + 1) // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch).numpy()
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()
            self.evaluate(it, loss/(it + 1))

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
