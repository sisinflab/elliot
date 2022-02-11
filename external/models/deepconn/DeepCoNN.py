"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from ast import literal_eval as make_tuple
from operator import itemgetter

from tqdm import tqdm
from .pointwise_pos_neg_sampler import Sampler
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .DeepCoNNModel import DeepCoNNModel

import numpy as np
import tensorflow as tf


class DeepCoNN(RecMixin, BaseRecommenderModel):
    r"""
    Joint Deep Modeling of Users and Items Using Reviews for Recommendation

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3018661.3018665>`_

    Args:
        batch_eval: Batch size for evaluation
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        u_rev_cnn_kernel: Tuple with kernel size for each user review cnn layer
        u_rev_cnn_features: Tuple with number of feature maps for each user review cnn layer
        i_rev_cnn_kernel: Tuple with kernel size for each item review cnn layer
        i_rev_cnn_features: Tuple with number of feature maps for each item review cnn layer
        latent_size: Latent size for the final fully-connected layer
        dropout: Dropout rate for each mlp layer in the model

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        DeepCoNN:
          meta:
            save_recs: True
          batch_eval: 64
          lr: 0.0005
          epochs: 50
          batch_size: 512
          l_w: 0.1
          u_rev_cnn_kernel: (3,)
          u_rev_cnn_features: (64,)
          i_rev_cnn_kernel: (3,)
          i_rev_cnn_features: (64,)
          latent_size: 128
          dropout: 0.5
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 64, int, None),
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_u_rev_cnn_kernel", "u_rev_cnn_kernel", "u_rev_cnn_kernel", "(3,)",
             lambda x: list(make_tuple(x)), None),
            ("_u_rev_cnn_features", "u_rev_cnn_features", "u_rev_cnn_features", "(64,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_i_rev_cnn_kernel", "i_rev_cnn_kernel", "i_rev_cnn_kernel", "(3,)",
             lambda x: list(make_tuple(x)), None),
            ("_i_rev_cnn_features", "i_rev_cnn_features", "i_rev_cnn_features", "(64,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_latent_size", "latent_size", "latent_size", 128, int, None),
            ("_dropout", "dropout", "dropout", 0.5, float, None),
            ("_loader", "loader", "loader", 'WordsTextualAttributes', str, None)
        ]
        self.autoset_params()

        self._interactions_textual = self._data.side_information.WordsTextualAttributes

        self._sampler = Sampler(self._data.i_train_dict,
                                self._data.public_users,
                                self._data.public_items,
                                self._interactions_textual.object.users_tokens,
                                self._interactions_textual.object.items_tokens)

        self._model = DeepCoNNModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            l_w=self._l_w,
            users_vocabulary_features=self._interactions_textual.object.users_word_features,
            items_vocabulary_features=self._interactions_textual.object.items_word_features,
            textual_words_feature_shape=self._interactions_textual.word_feature_shape,
            user_review_cnn_kernel=self._u_rev_cnn_kernel,
            user_review_cnn_features=self._u_rev_cnn_features,
            item_review_cnn_kernel=self._i_rev_cnn_kernel,
            item_review_cnn_features=self._i_rev_cnn_features,
            latent_size=self._latent_size,
            dropout_rate=self._dropout,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "DeepCoNN" \
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
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}

        users_tokens = self._sampler.users_tokens
        items_tokens = self._sampler.items_tokens

        self.logger.info('Starting convolutions for all users...')
        out_users = np.empty((self._num_users, self._latent_size))
        with tqdm(total=int(self._num_users // self._batch_eval), disable=not self._verbose) as t:
            for start_batch in range(0, self._num_users, self._batch_eval):
                stop_batch = min(start_batch + self._batch_eval, self._num_users)
                user_reviews = list(
                    itemgetter(*list(range(start_batch, stop_batch)))(users_tokens))
                out_users[start_batch: stop_batch] = self._model.conv_users(np.array(user_reviews, dtype=np.int64))
                t.update()
        self.logger.info('Convolutions for all users is complete!')

        self.logger.info('Starting convolutions for all items...')
        out_items = np.empty((self._num_items, self._latent_size))
        with tqdm(total=int(self._num_items // self._batch_eval), disable=not self._verbose) as t:
            for start_batch in range(0, self._num_items, self._batch_eval):
                stop_batch = min(start_batch + self._batch_eval, self._num_items)
                item_reviews = list(
                    itemgetter(*list(range(start_batch, stop_batch)))(items_tokens))
                out_items[start_batch: stop_batch] = self._model.conv_items(np.array(item_reviews, dtype=np.int64))
                t.update()
        self.logger.info('Convolutions for all items is complete!')

        self.logger.info('Starting predictions on all users/items pairs...')
        with tqdm(total=int(self._num_users // self._batch_eval), disable=not self._verbose) as t:
            for index, offset in enumerate(range(0, self._num_users, self._batch_eval)):
                offset_stop = min(offset + self._batch_eval, self._num_users)
                predictions = np.empty((offset_stop - offset, self._num_items))
                for item_index, item_offset in enumerate(range(0, self._num_items, self._batch_eval)):
                    item_offset_stop = min(item_offset + self._batch_eval, self._num_items)
                    p = self._model.predict(tf.Variable(out_users[offset: offset_stop], dtype=tf.float32),
                                            tf.Variable(out_items[item_offset: item_offset_stop],
                                                        dtype=tf.float32))
                    predictions[:, item_offset: item_offset_stop] = p
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)
                t.update()
        self.logger.info('Predictions on all users/items pairs is complete!')
        return predictions_top_k_val, predictions_top_k_test
