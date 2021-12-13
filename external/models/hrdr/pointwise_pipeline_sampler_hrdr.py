"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import tensorflow as tf

import numpy as np
import pandas as pd
import random
import os


class Sampler:
    def __init__(self, indexed_ratings, private_users, private_items, iu_dict, interactions_path, review_features_path, review_features_shape,
                 epochs):
        np.random.seed(42)
        random.seed(42)
        self._indexed_ratings = indexed_ratings
        self._private_users = private_users
        self._private_items = private_items
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._iu_dict = iu_dict
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

        self._review_features_path = review_features_path
        self._review_features_shape = review_features_shape
        self._interactions = pd.read_csv(interactions_path, sep='\t', header=None)
        self._epochs = epochs

    def load_reviews(self, u_re, i_re):
        # load user reviews
        user_reviews = np.empty((u_re.shape[0], self._review_features_shape))
        for idx in range(u_re.shape[0]):
            user_reviews[idx] = np.load(os.path.join(self._review_features_path, str(u_re[idx].numpy())) + '.npy')

        # load item reviews
        item_reviews = np.empty((i_re.shape[0], self._review_features_shape))
        for idx in range(i_re.shape[0]):
            item_reviews[idx] = np.load(os.path.join(self._review_features_path, str(i_re[idx].numpy())) + '.npy')

        return user_reviews, item_reviews

    def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        iu_dict = self._iu_dict
        lui_dict = self._lui_dict

        actual_inter = (events // batch_size) * batch_size * self._epochs

        counter_inter = 1

        def sample():
            # sample user/item pair
            u = r_int(n_users)
            u_pos = ui_dict[u]
            lui = lui_dict[u]
            if lui == n_items:
                sample()
            i = u_pos[r_int(lui)]
            i_pos = iu_dict[i]

            # get user ratings and item ratings
            u_ratings = np.zeros((1, self._nitems))
            i_ratings = np.zeros((self._nusers, 1))
            u_ratings[0, u_pos] = 1.0
            i_ratings[i_pos, 0] = 1.0

            # get user reviews and item reviews
            u_reviews = self._interactions[self._interactions[0] == self._private_users[u]][2]
            i_reviews = self._interactions[self._interactions[1] == self._private_items[i]][2]

            return u, i, u_pos, i_pos, u_ratings, i_ratings, u_reviews, i_reviews

        for ep in range(self._epochs):
            for _ in range(events):
                yield sample()
                if counter_inter == actual_inter:
                    return
                else:
                    counter_inter += 1

    def pipeline(self, num_users, batch_size):
        def load_func(u, i, u_p, i_p, u_ra, i_ra, u_re, i_re):
            user_reviews, item_reviews = tf.py_function(
                self.load_reviews,
                (u_re, i_re),
                (np.float32, np.float32)
            )
            return u, i, u_p, i_p, u_ra, i_ra, user_reviews, item_reviews

        data = tf.data.Dataset.from_generator(
            generator=self.step,
            output_types=(tf.int64, tf.int64, tf.int64, tf.int64, tf.float32, tf.float32, tf.int64, tf.int64),
            args=(num_users, batch_size)
        )
        data = data.map(load_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data
