"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

import tensorflow as tf
import os

import numpy as np
import random


class Sampler:
    def __init__(self, indexed_ratings, item_indices, cnn_features_path, epochs):
        np.random.seed(42)
        random.seed(42)
        self._indexed_ratings = indexed_ratings
        self._item_indices = item_indices
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

        self._cnn_features_path = cnn_features_path
        self._epochs = epochs

    def read_features_triple(self, user, pos, neg):
        # load positive and negative item features
        feat_pos = np.load(os.path.join(self._cnn_features_path, str(pos.numpy())) + '.npy')
        feat_neg = np.load(os.path.join(self._cnn_features_path, str(neg.numpy())) + '.npy')

        return user.numpy(), pos.numpy(), feat_pos, neg.numpy(), feat_neg

    def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict

        actual_inter = (events // batch_size) * batch_size * self._epochs

        counter_inter = 1

        def sample():
            u = r_int(n_users)
            ui = ui_dict[u]
            lui = lui_dict[u]
            if lui == n_items:
                sample()
            i = ui[r_int(lui)]

            j = r_int(n_items)
            while j in ui:
                j = r_int(n_items)
            return u, i, j

        for ep in range(self._epochs):
            for _ in range(events):
                yield sample()
                if counter_inter == actual_inter:
                    return
                else:
                    counter_inter += 1

    def pipeline(self, num_users, batch_size):
        def load_func(u, p, n):
            b = tf.py_function(
                self.read_features_triple,
                (u, p, n,),
                (np.int64, np.int64, np.float32, np.int64, np.float32)
            )
            return b

        data = tf.data.Dataset.from_generator(generator=self.step,
                                              output_shapes=((), (), ()),
                                              output_types=(tf.int64, tf.int64, tf.int64),
                                              args=(num_users, batch_size))
        data = data.map(load_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    def step_eval(self):
        for i_rel, i_abs in enumerate(self._item_indices):
            yield i_rel, i_abs

    # this is only for evaluation
    def pipeline_eval(self, batch_size):
        def load_func(i_r, i_a):
            b = tf.py_function(
                self.read_features,
                (i_r, i_a,),
                (np.int64, np.int64, np.float32)
            )
            return b

        data = tf.data.Dataset.from_generator(generator=self.step_eval,
                                              output_shapes=((), ()),
                                              output_types=(tf.int64, tf.int64))
        data = data.map(load_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    # this is only for evaluation
    def read_features(self, item_rel, item_abs):
        feat = np.load(os.path.join(self._cnn_features_path, str(item_abs.numpy())) + '.npy')

        return item_rel, item_abs, feat
