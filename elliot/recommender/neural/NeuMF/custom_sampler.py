"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import random

from elliot.dataset.samplers.base_sampler import TraditionalSampler


class Sampler(TraditionalSampler):
    def __init__(self, indexed_ratings, m, seed=42):
        super().__init__(seed, indexed_ratings)
        """np.random.seed(42)
        random.seed(42)
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}"""
        self._m = m

    def initialize(self):
        r_int = self._r_int
        n_items = self._nitems
        ui_dict = self._ui_dict
        pos = {(u, i, 1) for u, items in ui_dict.items() for i in items}

        neg = set()
        for u, i, _ in pos:
            ui = ui_dict[u]
            for _ in range(self._m):
                j = r_int(n_items)
                while j in ui:
                    j = r_int(n_items)
                neg.add((u, j, 0))

        samples = list(pos)
        samples.extend(list(neg))
        self._samples = random.sample(samples, len(samples))

    def _sample(self, bs, bsize):
        u, i, b = self._samples[bs:bs + bsize]
        return u, i, b

    #    for start in range(0, len(samples), batch_size):
    #        u, i, b = map(np.array, zip(*samples[start:min(start + batch_size, len(samples))]))
    #        yield u, i, b
