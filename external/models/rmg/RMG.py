from tqdm import tqdm
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .RMGModel import RMGModel
from .sampler import *

import numpy as np


class RMG(RecMixin, BaseRecommenderModel):
    r"""
    Reviews Meet Graphs: Enhancing User and Item Representations for Recommendation with Hierarchical Attentive Graph Neural Network

    For further details, please refer to the `paper <https://aclanthology.org/D19-1494/>`_
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_batch_eval", "batch_eval", "be", 256, int, None),
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_wcfm", "wcfm", "wcfm", 100, int, None),
            ("_wcfk", "wcfk", "wcfk", 3, int, None),
            ("_wa", "wa", "wa", 100, int, None),
            ("_scfm", "scfm", "scfm", 100, int, None),
            ("_scfk", "scfk", "scfk", 3, int, None),
            ("_sa", "sa", "sa", 100, int, None),
            ("_da", "da", "da", 100, int, None),
            ("_dau", "dau", "dau", 100, int, None),
            ("_factors", "factors", "f", 100, int, None),
            ("_uia", "uia", "uia", 100, int, None),
            ("_iua", "iua", "iua", 100, int, None),
            ("_ua", "ua", "ua", 100, int, None),
            ("_ia", "ia", "ia", 100, int, None),
            ("_dropout", "dropout", "d", 0.5, float, None),
            ("_loader", "loader", "l", 'WordsTextualAttributesPreprocessed', str, None)
        ]
        self.autoset_params()

        row, col = data.sp_i_train.nonzero()
        self.row = np.array(row)
        self.col = np.array(col)
        self.ratings = data.sp_i_train_ratings.data

        self._interactions_textual = self._data.side_information.WordsTextualAttributesPreprocessed
        self._interactions_textual.object.load_all_features()

        self._model = RMGModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            word_cnn_fea_maps=self._wcfm,
            word_cnn_fea_kernel=self._wcfk,
            word_att=self._wa,
            sent_cnn_fea_maps=self._scfm,
            sent_cnn_fea_kernel=self._scfk,
            sent_att=self._sa,
            doc_att=self._da,
            doc_att_u=self._dau,
            latent_size=self._factors,
            ui_att=self._uia,
            iu_att=self._iua,
            un_att=self._ua,
            in_att=self._ia,
            dropout_rate=self._dropout,
            max_reviews_user=self._interactions_textual.object.all_user_texts_shape[1],
            max_reviews_item=self._interactions_textual.object.all_item_texts_shape[1],
            max_sents=self._interactions_textual.object.all_user_texts_shape[2],
            max_sent_length=self._interactions_textual.object.all_user_texts_shape[3],
            max_neighbor=self._interactions_textual.object.user_to_item_shape[1],
            embed_vocabulary_features=self._interactions_textual.object.embed_vocabulary_features,
            random_seed=self._seed,
        )

    @property
    def name(self):
        return "RMG" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in generate_batch_data_random(self._interactions_textual.object.all_item_texts_features,
                                                        self._interactions_textual.object.all_user_texts_features,
                                                        self._interactions_textual.object.user_to_item_to_user_features,
                                                        self._interactions_textual.object.user_to_item_features,
                                                        self._interactions_textual.object.item_to_user_to_item_features,
                                                        self._interactions_textual.object.item_to_user_features,
                                                        self.col,
                                                        self.row,
                                                        self.ratings,
                                                        self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}

        self.logger.info('Starting pre-computation for all users...')
        out_users = np.empty((self._num_users, self._factors * 4))
        with tqdm(total=int(self._num_users // self._batch_eval), disable=not self._verbose) as t:
            for start_batch in range(0, self._num_users, self._batch_eval):
                stop_batch = min(start_batch + self._batch_eval, self._num_users)
                inputs = [
                    self._interactions_textual.object.all_user_texts_features[start_batch: stop_batch],
                    self._interactions_textual.object.user_to_item_to_user_features[start_batch: stop_batch],
                    self._interactions_textual.object.user_to_item_features[start_batch: stop_batch],
                    np.arange(start_batch, stop_batch)
                ]
                out_users[start_batch: stop_batch] = self._model.model_user(inputs, training=False)
                t.update()
        self.logger.info('Pre-computation for all users is complete!')

        self.logger.info('Starting pre-computation for all items...')
        out_items = np.empty((self._num_items, self._factors * 4))
        with tqdm(total=int(self._num_items // self._batch_eval), disable=not self._verbose) as t:
            for start_batch in range(0, self._num_items, self._batch_eval):
                stop_batch = min(start_batch + self._batch_eval, self._num_items)
                inputs = [
                    self._interactions_textual.object.all_item_texts_features[start_batch: stop_batch],
                    self._interactions_textual.object.item_to_user_to_item_features[start_batch: stop_batch],
                    self._interactions_textual.object.item_to_user_features[start_batch: stop_batch],
                    np.arange(start_batch, stop_batch)
                ]
                out_items[start_batch: stop_batch] = self._model.model_item(inputs, training=False)
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
                    inputs = [out_users[user_range], out_items[item_range]]
                    p = self._model.predict(inputs, offset_stop - offset, item_offset_stop - item_offset)
                    predictions[:, item_offset: item_offset_stop] = p.numpy()
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)
                t.update()
        self.logger.info('Predictions on all users/items pairs is complete!')
        return predictions_top_k_val, predictions_top_k_test
