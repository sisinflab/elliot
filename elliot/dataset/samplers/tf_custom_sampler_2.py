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
    def __init__(self, indexed_ratings=None, m=None, num_users=None, num_items=None, transactions=None, batch_size=512, random_seed=42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        self._UIDICT = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._POS = list({(u, i, 1) for u, items in self._UIDICT.items() for i in items})
        self._POS = random.sample(self._POS, len(self._POS))
        self._M = m
        self._NUM_USERS = num_users
        self._NUM_ITEMS = num_items
        self._transactions = transactions
        self._batch_size = batch_size

    def _full_generator(self):
        r_int = np.random.randint
        n_items = self._NUM_ITEMS
        ui_dict = self._UIDICT
        neg = set()
        for u, i, _ in self._POS:
            ui = ui_dict[u]
            for _ in range(self._M):
                j = r_int(n_items)
                while j in ui:
                    j = r_int(n_items)
                neg.add((u, j, 0))

        samples = self._POS[:]
        samples.extend(list(neg))
        samples = random.sample(samples, len(samples))

        # u, i, b = map(np.array, zip(*samples))
        # yield u,i,b
        for start in range(0, len(samples), self._batch_size):
            u, i, b = map(np.array, zip(*samples[start:min(start + self._batch_size, len(samples))]))
            yield u, i, b

    def step(self, batch_size: int):
        r_int = np.random.randint
        n_items = self._NUM_ITEMS
        ui_dict = self._UIDICT

        pos = {(u, i, 1) for u, items in ui_dict.items() for i in items}

        neg = set()
        for u, i, _ in pos:
            ui = ui_dict[u]
            for _ in range(self._M):
                j = r_int(n_items)
                while j in ui:
                    j = r_int(n_items)
                neg.add((u, j, 0))

        samples = list(pos)
        samples.extend(list(neg))
        samples = random.sample(samples, len(samples))

        for start in range(0, len(samples), batch_size):
            u, i, b = map(np.array, zip(*samples[start:min(start + batch_size, len(samples))]))
            yield u, i, b

    def create_tf_dataset(self):
        data = tf.data.Dataset.from_generator(generator=self._full_generator,
                                              output_types=(np.int64, np.int64, np.int64),
                                              )
        # data = data.unbatch().batch(batch_size=self._batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return data