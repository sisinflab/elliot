"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from tqdm import tqdm

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .ConvMFModel import convMF


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
            ("_factors_dim", "factors_dim", "fs", 200, int, None),
            ("_kernel_per_ws", "kernel_per_ws", "k_ws", 100, None, None),
            ("_give_item_weight", "give_item_weight", "g_iw", False, None, None),
            ("_loader", "loader", "load", "TextualAttributeSequence", None, None)
        ]
        self.autoset_params()
        self._side = getattr(self._data.side_information, self._loader, None)
        vocab_size = len(self._side.textual_features['X_vocab']) + 1
        CNN_X = self._side.textual_features['X_sequence']
        init_W = None

        if self._side.object.textual_feature_pretrain_path:
            init_W = self._side.load_word2vec_pretrain(self._embedding_size)

        self._iteration = 0
        if self._batch_size < 1:
            self._batch_size = self._data.num_users

        self._sp_i_train_ratings = self._data.sp_i_train_ratings

        self._model = convMF(self._data, self._lambda_u, self._lambda_i, self._embedding_size, self._factors_dim,
                             self._kernel_per_ws, self._drop_out_rate, self._epochs, vocab_size, self._max_len, CNN_X,
                             init_W, self._batch_size, self._give_item_weight, self._seed)

    @property
    def name(self):
        return "ConvMF" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        with tqdm(total=self._epochs, position=0, leave=True) as t:
            for it in self.iterate(self._epochs):
                loss = self._model.train_step()
                t.set_postfix({'loss': f'{loss / (it + 1):.5f}'})
                t.update()

                self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 10):
        self._model.prepare_predictions()

        return self._model.get_all_topks(self.get_candidate_mask(validation=True), k,
                                         self._data.private_users,
                                         self._data.private_items) \
                   if hasattr(self._data, "val_dict") \
                   else {}, self._model.get_all_topks(self.get_candidate_mask(), k,
                                                      self._data.private_users,
                                                      self._data.private_items)
