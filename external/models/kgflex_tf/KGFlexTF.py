import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
import threading
import tensorflow as tf

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin

from elliot.dataset.samplers import custom_sampler as cs

from .UserFeatureMapper import UserFeatureMapper
from .KGFlexTFModel import KGFlexTFModel
from .tfidf_utils import TFIDF

#mp.set_start_method('fork')


def uif_worker(us_f, its_f, mapping):
    uif = []
    lengths = []
    for it_f in its_f:
        s = set.intersection(set(map(lambda x: mapping[x], us_f)), it_f)
        lengths.append(len(s))
        uif.extend(list(s))
    return tf.RaggedTensor.from_row_lengths(uif, lengths)

def uif_worker2(queue_:mp.Queue, us_f, its_f, mapping):
    uif = []
    lengths = []
    for it_f in its_f:
        s = set.intersection(set(map(lambda x: mapping[x], us_f)), it_f)
        lengths.append(len(s))
        uif.extend(list(s))
    queue_.put(tf.RaggedTensor.from_row_lengths(uif, lengths))


class KGFlexTF(RecMixin, BaseRecommenderModel):

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        # auto parameters
        self._params_list = [
            ("_lr", "lr", "lr", 0.01, None, None),
            ("_embedding", "embedding", "em", 10, int, None),
            ("_first_order_limit", "first_order_limit", "fol", -1, None, None),
            ("_second_order_limit", "second_order_limit", "sol", -1, None, None),
            ("_l_w", "l_w", "l_w", 0.1, float, None),
            ("_l_b", "l_b", "l_b", 0.001, float, None),
            ("_loader", "loader", "load", "KGRec", None, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._side = getattr(self._data.side_information, self._loader, None)
        first_order_limit = self._params.first_order_limit
        second_order_limit = self._params.second_order_limit

        # ------------------------------ ITEM FEATURES ------------------------------
        print('importing items features')

        uri_to_private = {self._side.mapping[i]: p for i, p in self._data.public_items.items()}

        item_features_df1 = pd.DataFrame()
        item_features_df1['item'] = self._side.triples['uri'].map(uri_to_private)
        item_features_df1['f'] = list(zip(self._side.triples['predicate'], self._side.triples['object']))
        item_features_df1 = item_features_df1.dropna()
        self.item_features1 = item_features_df1.groupby('item')['f'].apply(set).to_dict()

        item_features_df2 = pd.DataFrame()
        item_features_df2['item'] = self._side.second_order_features['uri_x'].map(uri_to_private)
        item_features_df2['f'] = list(
            zip(self._side.second_order_features['predicate_x'], self._side.second_order_features['predicate_y'],
                self._side.second_order_features['object_y']))
        item_features_df2 = item_features_df2.dropna()
        self.item_features2 = item_features_df2.groupby('item')['f'].apply(set).to_dict()

        self.item_features = pd.concat([item_features_df1, item_features_df2]).groupby('item')['f'].apply(set).to_dict()



        print('finito tutto')

        # ------------------------------ USER FEATURES ------------------------------
        self.user_feature_mapper = UserFeatureMapper(data=self._data,
                                                     item_features=self.item_features1,
                                                     item_features2=self.item_features2,
                                                     first_order_limit=first_order_limit,
                                                     second_order_limit=second_order_limit)

        # ------------------------------ MODEL FEATURES ------------------------------
        self.logger.info('Features mapping started')
        users_features = self.user_feature_mapper.users_features
        features = set()
        for _, f in users_features.items():
            features = set.union(features, set(f))

        feature_key_mapping = dict(zip(list(features), range(len(features))))

        self.logger.info('FEATURES INFO: {} features found'.format(len(feature_key_mapping)))

        user_feature_weights = tf.constant(
            [[users_features[u].get(f, 0) for f in features] for u in self._data.private_users])

        item_features = list()
        for i in range(self._data.num_items):
            features = self.item_features[i]
            common = set.intersection(set(feature_key_mapping.keys()), features)
            # item_features[i] = list(map(lambda x: feature_key_mapping[x], common))
            item_features.append(list(map(lambda x: feature_key_mapping[x], common)))

        # for k, v in self.item_features.items():
        #     common = set.intersection(set(feature_key_mapping.keys()), v)
        #     item_features[k] = list(map(lambda x: feature_key_mapping[x], common))

        # RIMETTERE DA QUI PER TFIDF

        # self.tfidf_obj = TFIDF(item_features)
        # self.tfidf = self.tfidf_obj.tfidf()
        #
        # M = np.zeros((self._data.num_items, len(feature_key_mapping)))
        # for i in self.tfidf:
        #     M[tuple([i] * len(self.tfidf[i])), tuple(self.tfidf[i].keys())] = list(self.tfidf[i].values())

        def uif_args():
            return ((users_features[u],
                     item_features,
                     feature_key_mapping) for u in self._data.private_users.keys())

        arguments = uif_args()

        with mp.Pool(processes=mp.cpu_count()) as pool:
            user_item_features = pool.starmap(uif_worker, tqdm(arguments, total=len(self._data.private_users.keys()),
                                                               desc='User-Item Features'))

        # pool = mp.Pool(processes=mp.cpu_count())
        # user_item_features = pool.starmap(uif_worker, tqdm(arguments, total=len(self._data.private_users.keys()),
        #                                                    desc='User-Item Features'))
        # pool.join()
        # pool.close()


        # queue = mp.Queue()
        # p = mp.Process(target=uif_worker2, args=queue, arguments)

        # user_item_features = []
        # for u in tqdm(self._data.private_users.keys(), desc='User-Item Features'):
        #     user_item_features.append(uif_worker(users_features[u], item_features, feature_key_mapping))

        user_item_features = tf.ragged.stack(user_item_features)

        print('FATTO!')

        self._sampler = cs.Sampler(self._data.i_train_dict)

        # ------------------------------ MODEL ------------------------------
        self._model = KGFlexTFModel(num_users=self._data.num_users,
                                    num_items=self._data.num_items,
                                    user_feature_weights=user_feature_weights,
                                    user_item_features=user_item_features,
                                    num_features=len(feature_key_mapping),
                                    factors=self._embedding,
                                    learning_rate=self._lr)

    @property
    def name(self):
        return "KGFlexTF" \
               + "_e:" + str(self._epochs) \
               + f"_{self.get_params_shortcut()}"

    def get_recommendations(self, k: int = 10):
        predictions = self._model.get_all_recs()
        return self._model.get_all_topks(predictions, self.get_candidate_mask(validation=True), k,
                                         self._data.private_users, self._data.private_items) if hasattr(self._data,
                                                                                                        "val_dict") else {}, self._model.get_all_topks(
            predictions, self.get_candidate_mask(), k, self._data.private_users, self._data.private_items)

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