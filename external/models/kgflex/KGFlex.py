import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.dataset.samplers import custom_sampler as cs

from .UserFeatureMapper import UserFeatureMapper
from .UserFeatureMapper2 import UserFeatureMapper2
from collections import defaultdict
from .kgflexmodel import KGFlexModel

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
        np.random.seed(self._seed)
        self._side = getattr(self._data.side_information, self._loader, None)

        # ------------------------------ ITEM FEATURES ------------------------------
        print('importing items features')
        # pd.merge(self._side.triples, self._side.triples, left_on='object', right_on='uri', how='left')
        self.item_features = {item: set(map(tuple,
                                            self._side.triples[
                                                       self._side.triples.uri ==
                                                       self._side.mapping[self._data.private_items[item]]]
                                                   [['predicate', 'object']].values))
                              for item in self._data.private_items}

        # ------------------------------ USER FEATURES ------------------------------
        print('user features loading')

        self.user_feature_mapper = UserFeatureMapper2(data=self._data,
                                                      item_features=self.item_features)

        self.user_feature_mapper = UserFeatureMapper(self._data.i_train_dict,
                                                     self.item_features,
                                                     self._side.mapping, self._seed)
        client_ids = list(self._data.i_train_dict.keys())

        self.user_feature_mapper.compute_and_export_features(client_ids, self._parallel_ufm, self._first_order_limit,
                                                             self._second_order_limit)

        # ------------------------------ MODEL FEATURES ------------------------------
        print('features mapping')
        users_features = self.user_feature_mapper.client_features

        features = set()
        for c in client_ids:
            features = set.union(features, users_features[c])
        feature_key_mapping = dict(zip(list(features), range(len(features))))

        # mapping features in columns
        features_mapping = defaultdict(lambda: len(features_mapping))
        for c in client_ids:
            for feature in users_features[c]:
                _ = features_mapping[feature]

        # total number of features (i.e. columns of the item matrix / latent factors)
        print('FEATURES INFO: {} features found'.format(len(features_mapping)))
        item_features_mask = []
        for _, v in self.item_features.items():
            common = set.intersection(set(features_mapping.keys()), set(v))
            item_features_mask.append([True if f in common else False for f in features_mapping])
        self.item_features_mask = csr_matrix(item_features_mask)

        index_mask = {user: [True if f in users_features[user] else False
                             for f in features_mapping] for user in self._data.privateusers.keys()}

        # ------------------------------ POSITIVE AND NEGATIVE ITEMS ------------------------------

        self._sampler = cs.Sampler(self._data.i_train_dict)

        # ------------------------------ MODEL ------------------------------

        self._model = KGFlexModel(learning_rate=self._lr,
                                  n_users=self._data.num_users,
                                  users=self._data.privateusers.keys(),
                                  n_items=self._data.num_items,
                                  n_features=len(features_mapping),
                                  feature_key_mapping=feature_key_mapping,
                                  item_features_mapper=self.item_features,
                                  embedding_size=self._embedding,
                                  index_mask=index_mask,
                                  users_features=users_features,
                                  data=self._data)


    @property
    def name(self):
        return "KGFlex" \
               + "_e:" + str(self._epochs) \
               + f"_{self.get_params_shortcut()}"

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_recs(u, mask, k) for u in self._data.users}

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
