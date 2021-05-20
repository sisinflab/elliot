"""
Module description:
"""
__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'
import tensorflow as tf
import numpy as np


class Sampler(tf.data.Dataset):
    def _pos_generator(self):
        ui_dict = self._ui_dict
        pos = {(u, i, 1) for u, items in ui_dict.items() for i in items}
        while True:
            for u, i, _ in pos:
                yield u, i

    def _generator(self, batch_size: int):
        r_int = np.random.randint
        n_items = self._nitems
        ui_dict = self._ui_dict

        for _ in range(batch_size):
            u, i = next(self._pos)
            ui = ui_dict[u]
            for _ in range(self._m):
                j = r_int(n_items)
                while j in ui:
                    j = r_int(n_items)
                yield u, i, j

    def __new__(cls, indexed_ratings=None, m=1, batch_size=512, random_seed=42):
        np.random.seed(random_seed)
        data = tf.data.Dataset.from_generator(generator=cls._generator,
                                              output_shapes=((), (), ()),
                                              output_types=(tf.int64, tf.int64, tf.int64),
                                              args=(batch_size,))
        data = data.batch(batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        data = data.prefetch(buffer_size=tf.data.AUTOTUNE)
        data._indexed_ratings = indexed_ratings
        data._users = list(data._indexed_ratings.keys())
        data._nusers = len(data._users)
        data._items = list({k for a in data._indexed_ratings.values() for k in a.keys()})
        data._nitems = len(data._items)
        data._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        data._lui_dict = {u: len(v) for u, v in data._ui_dict.items()}
        data._m = m

        data._pos = data._pos_generator()
        return data