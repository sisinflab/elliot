"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import math
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .ConvMFModel import convMF

np.random.seed(42)


class ConvMF(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._random = np.random

        self._ratings = self._data.train_dict

        # autoset params
        self._params_list = [
            ("_lambda_u", "l_u", "lu", 1e-5, None, None),
            ("_lambda_i", "l_i", "li", 1e-5, None, None),
            ("_drop_out_rate", "drop_out", "dor", 0.2, None, None),
            ("_max_len", "max_length", "max_len", 300, int, None),
            ("_embedding_size", "embedding_size", "es", 200, int, None),
            ("_factors_dim", "factors_dim", "fs", 50, int, None),
            ("_kernel_per_ws", "kernel_per_ws", "k_ws", 100, None, None),
            ("_loader", "loader", "load", "TextualAttributeSequence", None, None)
        ]
        self.autoset_params()
        self._side = getattr(self._data.side_information, self._loader, None)
        vocab_size = len(self._side.textual_features['X_vocab']) + 1
        CNN_X = self._side.textual_features['X_sequence']

        self._iteration = 0
        if self._batch_size < 1:
            self._batch_size = self._data.num_users

        self._sp_i_train_ratings = self._data.sp_i_train_ratings

        self._model = convMF(self._lambda_u, self._lambda_i, self._embedding_size, self._factors_dim,
                             self._kernel_per_ws, self._drop_out_rate, self._epochs, self._sp_i_train_ratings,
                             vocab_size, self._max_len, CNN_X, self._seed)

    @property
    def name(self):
        return "ConvMF" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0

            if it % 10 < self._step_to_switch:
                with tqdm(total=int(self._epoch_length // self._batch_size), disable=not self._verbose) as t:
                    for batch in self._sampler.step(self._batch_size):
                        steps += 1
                        loss += self._model.train_step_rec(batch, is_rec=True)
                        t.set_postfix({'loss REC': f'{loss.numpy() / steps:.5f}'})
                        t.update()
            else:
                with tqdm(total=int(self._epoch_length // self._batch_size), disable=not self._verbose) as t:
                    for batch in self._triple_sampler.step(self._batch_size):
                        steps += 1
                        loss += self._model.train_step_kg(batch, is_rec=False, kg_lambda=self._kg_lambda)
                        t.set_postfix({'loss KGC': f'{loss.numpy() / steps:.5f}'})
                        t.update()

            self.evaluate(it, loss / (it + 1))

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
