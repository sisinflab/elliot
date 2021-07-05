"""
Module description:
"""
__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'
import tensorflow as tf
import numpy as np
import random


class Sampler():
    def __init__(self, indexed_ratings, m, transactions, random_seed=42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        self._transactions = transactions
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        self._m = m
        self._pos = self._pos_generator(self._ui_dict)

    @staticmethod
    def _pos_generator(ui_dict):
        # ui_dict = self._ui_dict
        pos = {(u, i, 1) for u, items in ui_dict.items() for i in items}
        while True:
            for u, i, _ in pos:
                yield u, i

    # @staticmethod
    def _generator(self, num_samples: int):
        r_int = np.random.randint
        n_items = self._nitems
        ui_dict = self._ui_dict

        for _ in range(num_samples):
            u, i = next(self._pos)
            ui = ui_dict[u]
            for _ in range(self._m):
                j = r_int(n_items)
                while j in ui:
                    j = r_int(n_items)
                yield u, i, j

    def create_dataset(self, batch_size=512, random_seed=42):

        data = tf.data.Dataset.from_generator(generator=self._generator,
                                              output_shapes=((), (), ()),
                                              output_types=(tf.int64, tf.int64, tf.int64),
                                              args=(self._transactions * self._m,))
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # data._indexed_ratings = indexed_ratings
        # data._users = list(data._indexed_ratings.keys())
        # data._nusers = len(data._users)
        # data._items = list({k for a in data._indexed_ratings.values() for k in a.keys()})
        # data._nitems = len(data._items)
        # data._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        # data._lui_dict = {u: len(v) for u, v in data._ui_dict.items()}
        # data._m = m
        # data._pos_generator = cls._pos_generator
        # data._pos = self._pos_generator(data._ui_dict)
        return data