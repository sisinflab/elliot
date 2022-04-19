from ast import literal_eval as make_tuple
from operator import itemgetter

from tqdm import tqdm
from .pointwise_pos_neg_sampler import Sampler
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .NARREModel import NARREModel

import numpy as np


class NARRE(RecMixin, BaseRecommenderModel):
    r"""
    Neural Attentional Rating Regression with Review-level Explanations
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
            ("_u_rev_cnn_k", "u_rev_cnn_k", "u_rev_cnn_k", "(3,)",
             lambda x: list(make_tuple(x)), None),
            ("_u_rev_cnn_f", "u_rev_cnn_f", "u_rev_cnn_f", 100, int, None),
            ("_i_rev_cnn_k", "i_rev_cnn_k", "i_rev_cnn_k", "(3,)",
             lambda x: list(make_tuple(x)), None),
            ("_i_rev_cnn_f", "i_rev_cnn_f", "i_rev_cnn_f", 100, int, None),
            ("_att_u", "att_u", "att_u", 100, int, None),
            ("_att_i", "att_i", "att_i", 100, int, None),
            ("_lat_s", "lat_s", "lat_s", 100, int, None),
            ("_n_lat", "n_lat", "n_lat", 100, int, None),
            ("_pretr", "pretr", "pretr", True, bool, None),
            ("_dropout", "dropout", "dropout", 0.5, float, None),
            ("_loader", "loader", "loader", 'WordsTextualAttributes', str, None)
        ]
        self.autoset_params()

        np.random.seed(self._seed)

        self._interactions_textual = self._data.side_information.WordsTextualAttributes

        self._sampler = Sampler(
            self._data.i_train_dict,
            self._data.public_users,
            self._data.public_items,
            self._interactions_textual.object.users_tokens,
            self._interactions_textual.object.items_tokens,
            self._interactions_textual.object.pos_users,
            self._interactions_textual.object.pos_items
        )

        self._model = NARREModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            l_w=self._l_w,
            users_vocabulary_features=self._interactions_textual.object.users_word_features,
            items_vocabulary_features=self._interactions_textual.object.items_word_features,
            textual_words_feature_shape=self._interactions_textual.word_feature_shape,
            user_review_cnn_kernel=self._u_rev_cnn_k,
            user_review_cnn_features=self._u_rev_cnn_f,
            item_review_cnn_kernel=self._i_rev_cnn_k,
            item_review_cnn_features=self._i_rev_cnn_f,
            attention_size_user=self._att_u,
            attention_size_item=self._att_i,
            latent_size=self._lat_s,
            n_latent=self._n_lat,
            dropout_rate=self._dropout,
            pretrained=self._pretr,
            random_seed=self._seed,
        )

    @property
    def name(self):
        return "NARRE" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        row, col = self._data.sp_i_train.nonzero()
        ratings = self._data.sp_i_train_ratings.data
        edge_index = np.array([row, col, ratings]).transpose()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            np.random.shuffle(edge_index)
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(edge_index, self._data.transactions, self._batch_size):
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
        pos_users = self._sampler.pos_users
        pos_items = self._sampler.pos_items

        self.logger.info('Starting pre-computation for all users...')
        out_users = np.empty((self._num_users, self._n_lat))
        with tqdm(total=int(self._num_users // self._batch_eval), disable=not self._verbose) as t:
            for start_batch in range(0, self._num_users, self._batch_eval):
                stop_batch = min(start_batch + self._batch_eval, self._num_users)
                user_reviews = list(
                    itemgetter(*list(range(start_batch, stop_batch)))(users_tokens))
                user_pos = list(
                    itemgetter(*list(range(start_batch, stop_batch)))(pos_users))
                inputs = [np.arange(start_batch, stop_batch),
                          np.array(user_reviews, dtype=np.int64),
                          np.array(user_pos, dtype=np.int64)]
                out_users[start_batch: stop_batch] = self._model.forward_user_embeddings(inputs, training=False)
                t.update()
        self.logger.info('Pre-computation for all users is complete!')

        self.logger.info('Starting pre-computation for all items...')
        out_items = np.empty((self._num_items, self._n_lat))
        with tqdm(total=int(self._num_items // self._batch_eval), disable=not self._verbose) as t:
            for start_batch in range(0, self._num_items, self._batch_eval):
                stop_batch = min(start_batch + self._batch_eval, self._num_items)
                item_reviews = list(
                    itemgetter(*list(range(start_batch, stop_batch)))(items_tokens))
                item_pos = list(
                    itemgetter(*list(range(start_batch, stop_batch)))(pos_items))
                inputs = [np.arange(start_batch, stop_batch),
                          np.array(item_reviews, dtype=np.int64),
                          np.array(item_pos, dtype=np.int64)]
                out_items[start_batch: stop_batch] = self._model.forward_item_embeddings(inputs, training=False)
                t.update()
        self.logger.info('Pre-computation for all items is complete!')

        self.logger.info('Starting predictions on all users/items pairs...')
        with tqdm(total=int(self._num_users // self._batch_eval), disable=not self._verbose) as t:
            for index, offset in enumerate(range(0, self._num_users, self._batch_eval)):
                offset_stop = min(offset + self._batch_eval, self._num_users)
                predictions = np.empty((offset_stop - offset, self._num_items))
                for item_index, item_offset in enumerate(range(0, self._num_items, self._batch_eval)):
                    item_offset_stop = min(item_offset + self._batch_eval, self._num_items)
                    user_range = np.repeat(np.arange(offset, offset_stop), repeats=item_offset_stop - item_offset)
                    item_range = np.tile(np.arange(item_offset, item_offset_stop), reps=offset_stop - offset)
                    p = self._model.predict(out_users[user_range],
                                            user_range,
                                            out_items[item_range],
                                            item_range,
                                            offset_stop - offset,
                                            item_offset_stop - item_offset)
                    predictions[:, item_offset: item_offset_stop] = p.numpy()
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)
                t.update()
        self.logger.info('Predictions on all users/items pairs is complete!')
        return predictions_top_k_val, predictions_top_k_test
