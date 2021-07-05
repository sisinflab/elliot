"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

import tensorflow as tf

import numpy as np
import random
import os


class Sampler:
    def __init__(self, indexed_ratings, cnn_features_path, cnn_features_shape, epochs):
        np.random.seed(42)
        random.seed(42)
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

        self._cnn_features_path = cnn_features_path
        self._cnn_features_shape = cnn_features_shape
        self._epochs = epochs

    def read_features_triple(self, user, pos, neg, user_pos):
        # load positive and negative item features
        item_pos = np.empty((user_pos.shape[0], *self._cnn_features_shape))
        for idx in range(user_pos.shape[0]):
            item_pos[idx] = np.load(os.path.join(self._cnn_features_path, str(user_pos[idx].numpy())) + '.npy')

        return user.numpy(), pos.numpy(), neg.numpy(), user_pos.numpy(), item_pos

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
            return u, i, j, ui

        for ep in range(self._epochs):
            for _ in range(events):
                yield sample()
                if counter_inter == actual_inter:
                    return
                else:
                    counter_inter += 1

    def pipeline(self, num_users, batch_size):
        def load_func(u, p, n, up):
            b = tf.py_function(
                self.read_features_triple,
                (u, p, n, up),
                (np.int64, np.int64, np.int64, np.int64, np.float32)
            )
            return b
        data = tf.data.Dataset.from_generator(generator=self.step,
                                              output_shapes=((), (), (), (None,)),
                                              output_types=(tf.int64, tf.int64, tf.int64, tf.int64),
                                              args=(num_users, batch_size))
        data = data.map(load_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    def step_eval(self):
        n_users = self._nusers
        ui_dict = self._ui_dict

        for u in range(n_users):
            yield u, ui_dict[u]

    # this is only for evaluation
    def pipeline_eval(self):
        def load_func(u, up):
            b = tf.py_function(
                self.read_features_eval,
                (u, up),
                (np.int64, np.int64, np.float32)
            )
            return b

        data = tf.data.Dataset.from_generator(generator=self.step_eval,
                                              output_shapes=((), (None,)),
                                              output_types=(tf.int64, tf.int64))
        data = data.map(load_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    # this is only for evaluation
    def read_features_eval(self, user, user_pos):
        item = np.empty((user_pos.shape[0], *self._cnn_features_shape))
        for idx in range(user_pos.shape[0]):
            item[idx] = np.load(os.path.join(self._cnn_features_path, str(user_pos[idx].numpy())) + '.npy')

        return user.numpy(), user_pos.numpy(), item
