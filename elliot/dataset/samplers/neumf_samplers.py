"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import random
from tqdm import tqdm

from elliot.dataset.samplers.base_sampler import AbstractSampler


class NeuMFSampler(AbstractSampler):
    def __init__(self, m, **params):
        super().__init__(**params)
        """np.random.seed(42)
        random.seed(42)
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}"""
        self.m = m
        self.events = self.events * (self.m + 1)

    def _sample(self, **kwargs):
        r_int = self._r_int
        n_items = self._nitems
        ui_dict = self._ui_dict
        pos = {(u, i, 1) for u, items in ui_dict.items() for i in items}

        iter_data = tqdm(pos, total=len(pos), desc="Sampling")

        neg = set()
        for u, i, _ in iter_data:
            ui = ui_dict[u]
            for _ in range(self.m):
                j = r_int(n_items)
                while j in ui:
                    j = r_int(n_items)
                neg.add((u, j, 0))

        samples = list(pos)
        samples.extend(list(neg))
        samples = random.sample(samples, len(samples))
        u, i, b = map(np.array, zip(*samples))
        return u, i, b

    # def _sample(self, bs, bsize):
    #     u, i, b = map(np.array, zip(*self._samples[bs:bs + bsize]))
    #     return u, i, b

    #    for start in range(0, len(samples), batch_size):
    #        u, i, b = map(np.array, zip(*samples[start:min(start + batch_size, len(samples))]))
    #        yield u, i, b


class CustomNeuMFSampler(AbstractSampler):
    def __init__(self, m, **params):
        super().__init__(**params)
        #np.random.seed(random_seed)
        #random.seed(random_seed)
        """elf._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}"""
        self.m = m
        self.events = self.events * (self.m + 1)
        self._pos = self._pos_generator(self._ui_dict)

    @staticmethod
    def _pos_generator(ui_dict):
        # ui_dict = self._ui_dict
        pos = {(u, i, 1) for u, items in ui_dict.items() for i in items}
        while True:
            for u, i, _ in pos:
                yield u, i

    def _sample(self, **kwargs):
        users, pos, neg = [], [], []

        r_int = self._r_int
        n_items = self._nitems
        ui_dict = self._ui_dict

        iter_data = tqdm(range(self.events), total=self.events, desc="Sampling")

        for _ in iter_data:
            u, i = next(self._pos)
            ui = ui_dict[u]
            for _ in range(self.m + 1):
                j = r_int(n_items)
                while j in ui:
                    j = r_int(n_items)
                users.append(u)
                pos.append(i)
                neg.append(j)

        return users, pos, neg

    # @staticmethod
    # def _sample(self, **kwargs):
    #     r_int = self._r_int
    #     n_items = self._nitems
    #     ui_dict = self._ui_dict
    #
    #     for _ in range(self.events):
    #         u, i = next(self._pos)
    #         ui = ui_dict[u]
    #         for _ in range(self.m):
    #             j = r_int(n_items)
    #             while j in ui:
    #                 j = r_int(n_items)
    #             yield u, i, j

    """def create_dataset(self, batch_size=512, random_seed=42):

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
        return data"""