import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pandas as pd

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.dataset.samplers import custom_sampler as cs

from .UserFeatureMapper import UserFeatureMapper
from .KGFlexModel import KGFlexModel


class KGFlex(RecMixin, BaseRecommenderModel):

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        # auto parameters
        self._params_list = [
            ("_lr", "lr", "lr", 0.01, None, None),
            ("_embedding", "embedding", "em", 10, int, None),
            ("_first_order_limit", "first_order_limit", "fol", -1, None, None),
            ("_second_order_limit", "second_order_limit", "sol", -1, None, None),
            ("_loader", "loader", "load", "KGRec", None, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._side = getattr(self._data.side_information, self._loader, None)
        first_order_limit = self._params.first_order_limit
        second_order_limit = self._params.second_order_limit
        embedding = self._embedding

        # ------------------------------ ITEM FEATURES ------------------------------
        # self.item_features = {item: set(map(tuple,
        #                                     self._side.triples[
        #                                         self._side.triples.uri ==
        #                                         self._side.mapping[self._data.private_items[item]]]
        #                                     [['predicate', 'object']].values))
        #                       for item in tqdm(self._data.private_items, desc='first order features')}
        #
        # self.item_features2 = dict()
        # sof = self._side.second_order_features
        # for item in tqdm(self._data.private_items, desc='second order features'):
        #     item_f = sof[sof.uri_x == self._side.mapping[self._data.private_items[item]]][
        #         ['predicate_x', 'predicate_y', 'object_y']]
        #     self.item_features2[item] = {(f'<{p1}><{p2}>', o) for p1, p2, o in item_f.values}

        uri_to_private = {v: self._data.public_items[k] for k, v in self._side.mapping.items()}
        self.logger.info('Item features extraction...')
        item_features_df = pd.DataFrame()
        item_features_df['item'] = self._side.triples['uri'].map(uri_to_private)
        item_features_df['f'] = list(zip(self._side.triples['predicate'], self._side.triples['object']))
        item_features_df = item_features_df.dropna().astype({'item': int})
        self.item_features1 = item_features_df.groupby('item')['f'].apply(set).to_dict()

        item_features_df = pd.DataFrame()
        item_features_df['item'] = self._side.second_order_features['uri_x'].map(uri_to_private)
        item_features_df['f'] = list(
            zip(self._side.second_order_features['predicate_x'], self._side.second_order_features['predicate_y'],
                self._side.second_order_features['object_y']))
        item_features_df = item_features_df.dropna().astype({'item': int})
        self.item_features2 = item_features_df.groupby('item')['f'].apply(set).to_dict()

        # ------------------------------ USER FEATURES ------------------------------
        self.logger.info('FEATURES INFO: user features selection...')
        self.user_feature_mapper = UserFeatureMapper(data=self._data,
                                                     item_features=self.item_features1,
                                                     item_features2=self.item_features2,
                                                     first_order_limit=first_order_limit,
                                                     second_order_limit=second_order_limit)

        # ------------------------------ MODEL FEATURES ------------------------------
        self.logger.info('Features mapping started')

        features = set()
        users_features = self.user_feature_mapper.users_features
        for _, f in users_features.items():
            features = set.union(features, set(f))

        item_features = {item: set.intersection(set.union(self.item_features1.get(item, {}),
            self.item_features2.get(item, {})), features) for item in self._data.private_items}

        feature_key_mapping = dict(zip(features, range(len(features))))

        self.logger.info('FEATURES INFO: {} features found'.format(len(features)))

        users_features_mask = {user: [True if f in users_features[user] else False
                                      for f in feature_key_mapping] for user in self._data.private_users.keys()}

        self._sampler = cs.Sampler(self._data.i_train_dict)

        # ------------------------------ MODEL ------------------------------
        self._model = KGFlexModel(learning_rate=self._lr,
                                  n_users=self._data.num_users,
                                  n_items=self._data.num_items,
                                  n_features=len(features),
                                  feature_key_mapping=feature_key_mapping,
                                  item_features_mapper=item_features,
                                  embedding_size=embedding,
                                  users_features=users_features,
                                  data=self._data)

    @property
    def name(self):
        return "KGFlex" \
               + "_e:" + str(self._epochs) \
               + f"_{self.get_params_shortcut()}"

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_recs(u, mask, k) for u in tqdm(self._data.users)}

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)
        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

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
            self.evaluate(it, loss)
