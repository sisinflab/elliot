"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from ast import literal_eval as make_tuple

from operator import itemgetter

from tqdm import tqdm
from .pointwise_pos_neg_sampler_hrdr import Sampler
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .HRDRModel import HRDRModel

import numpy as np
import tensorflow as tf


class HRDR(RecMixin, BaseRecommenderModel):
    r"""
    Hybrid neural recommendation with joint deep representation learning of ratings and reviews

    For further details, please refer to the `paper <https://www.sciencedirect.com/science/article/pii/S0925231219313207>`_

    Args:
        batch_eval: Batch size for evaluation
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        u_proj_rating: Tuple with number of units for each user projection rating layer
        i_proj_rating: Tuple with number of units for each item projection rating layer
        u_rev_cnn: Tuple with number of feature maps for each user review cnn layer
        i_rev_cnn: Tuple with number of feature maps for each item review cnn layer
        u_rev_att: Tuple with number of units for each user attention layer
        i_rev_att: Tuple with number of units for each item attention layer
        u_fin_rep: Tuple with number of units for each user final representation layer
        i_fin_rep: Tuple with number of units for each item final representation layer
        dropout: Dropout rate for each mlp layer in the model

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        HRDR:
          meta:
            save_recs: True
          batch_eval: 64
          lr: 0.0005
          epochs: 50
          batch_size: 512
          factors: 64
          l_w: 0.1
          u_proj_rating: (64,)
          i_proj_rating: (64,)
          u_rev_cnn: (64,)
          i_rev_cnn: (64,)
          u_rev_att: (64,)
          i_rev_att: (64,)
          u_fin_rep: (64,)
          i_fin_rep: (64,)
          dropout: 0.5
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._iu_dict = data.build_items_neighbour()

        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 64, int, None),
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_u_proj_rating", "u_proj_rating", "u_proj_rating", "(64,)",
             lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_i_proj_rating", "i_proj_rating", "i_proj_rating", "(64,)",
             lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_u_rev_cnn", "u_rev_cnn", "u_rev_cnn", "(64,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_i_rev_cnn", "i_rev_cnn", "i_rev_cnn", "(64,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_u_rev_att", "u_rev_att", "u_rev_att", "(64,)",
             lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_i_rev_att", "i_rev_att", "i_rev_att", "(64,)",
             lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_u_fin_rep", "u_fin_rep", "u_fin_rep", "(64,)",
             lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_i_fin_rep", "i_fin_rep", "i_fin_rep", "(64,)",
             lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_dropout", "dropout", "dropout", 0.5, float, None),
            ("_loader", "loader", "loader", 'WordsTextualAttributes', str, None)
        ]
        self.autoset_params()

        self._interactions_textual = self._data.side_information.WordsTextualAttributes

        self._pad_index = self._interactions_textual.object.word_features.shape[0] - 1

        self._ui_dict = {u: list(set(self._data.i_train_dict[u])) for u in self._data.i_train_dict}

        self._sampler = Sampler(self._ui_dict,
                                self._iu_dict,
                                self._data.public_users,
                                self._data.public_items,
                                self._interactions_textual.object.users_tokens,
                                self._interactions_textual.object.items_tokens)

        self._model = HRDRModel(
            num_users=self._num_users,
            num_items=self._num_items,
            batch_size=self._batch_size,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            l_w=self._l_w,
            vocabulary_features=self._interactions_textual.object.word_features,
            user_projection_rating=self._u_proj_rating,
            item_projection_rating=self._i_proj_rating,
            user_review_cnn=self._u_rev_cnn,
            item_review_cnn=self._i_rev_cnn,
            user_review_attention=self._u_rev_att,
            item_review_attention=self._i_rev_att,
            user_final_representation=self._u_fin_rep,
            item_final_representation=self._i_fin_rep,
            dropout=self._dropout,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "HRDR" \
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

        self.logger.info('Starting all operations for users...')
        xu = np.empty((self._num_users, self._model.user_projection_rating[-1]))
        ou = np.empty((self._num_users, self._model.user_final_representation[-1]))
        with tqdm(total=int(self._num_users // self._batch_eval), disable=not self._verbose) as t:
            for start_batch in range(0, self._num_users, self._batch_eval):
                stop_batch = min(start_batch + self._batch_eval, self._num_users)
                u_ratings = self._data.sp_i_train.todense[start_batch: stop_batch]
                xu[start_batch: stop_batch] = self._model.user_projection_rating_network(u_ratings, training=False)
                user_reviews = list(itemgetter(*list(range(start_batch, stop_batch)))(users_tokens))
                user_reviews_features = tf.nn.embedding_lookup(self._model.V, user_reviews)
                ou_current = tf.reduce_max(self._model.user_review_cnn_network(user_reviews_features), axis=-2)
                qru = self._model.user_review_attention_network(xu, training=False)
                au = tf.reduce_sum(tf.multiply(ou_current, qru), axis=-1, keepdims=True)
                au_norm = tf.nn.softmax(au, axis=1)
                ou_current = tf.multiply(ou_current, au_norm)
                ou_current = tf.reduce_sum(ou_current, 1)
                ou[start_batch: stop_batch] = self._model.user_final_representation_network(ou_current, training=False)
                t.update()
        self.logger.info('Operations for users are complete!')

        self.logger.info('Starting all operations for items...')
        xi = np.empty((self._num_items, self._model.item_projection_rating[-1]))
        oi = np.empty((self._num_items, self._model.item_final_representation[-1]))
        with tqdm(total=int(self._num_items // self._batch_eval), disable=not self._verbose) as t:
            for start_batch in range(0, self._num_items, self._batch_eval):
                stop_batch = min(start_batch + self._batch_eval, self._num_items)
                i_ratings = self._data.sp_i_train.todense[:, start_batch: stop_batch]
                xi[start_batch: stop_batch] = self._model.item_projection_rating_network(i_ratings, training=False)
                item_reviews = list(itemgetter(*list(range(start_batch, stop_batch)))(items_tokens))
                item_reviews_features = tf.nn.embedding_lookup(self._model.V, item_reviews)
                oi_current = tf.reduce_max(self._model.item_review_cnn_network(item_reviews_features), axis=-2)
                qri = self._model.item_review_attention_network(xi, training=False)
                ai = tf.reduce_sum(tf.multiply(oi_current, qri), axis=-1, keepdims=True)
                ai_norm = tf.nn.softmax(ai, axis=1)
                oi_current = tf.multiply(oi_current, ai_norm)
                oi_current = tf.reduce_sum(oi_current, 1)
                oi[start_batch: stop_batch] = self._model.item_final_representation_network(oi_current, training=False)
                t.update()
        self.logger.info('Operations for items are complete!')

        for index, offset in enumerate(range(0, self._num_users, self._batch_eval)):
            offset_stop = min(offset + self._batch_eval, self._num_users)
            predictions = np.empty((offset_stop - offset, self._num_items))
            with tqdm(total=int(self._num_items // self._batch_eval), disable=not self._verbose) as t:
                for item_index, item_offset in enumerate(range(0, self._num_items, self._batch_eval)):
                    item_offset_stop = min(item_offset + self._batch_eval, self._num_items)
                    p = self._model.predict(list(range(offset, offset_stop)),
                                            list(range(item_offset, item_offset_stop)),
                                            tf.Variable(xu[offset: offset_stop], tf.float32),
                                            tf.Variable(xi[item_offset: item_offset_stop], tf.float32),
                                            tf.Variable(ou[offset: offset_stop], tf.float32),
                                            tf.Variable(oi[item_offset: item_offset_stop], tf.float32))
                    predictions[:(offset_stop - offset), item_index * self._batch_eval:item_offset_stop] = p
                    t.update()
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test
