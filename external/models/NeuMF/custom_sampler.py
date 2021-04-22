"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
np.random.seed(42)
import random
random.seed(42)


class Sampler:
    def __init__(self, indexed_ratings, m):
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        self._m = m

    def step(self, batch_size: int):
        r_int = np.random.randint
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict

        def sample_pos(u):
            ui = ui_dict[u]
            lui = lui_dict[u]
            if lui == n_items:
                return None
            return ui[r_int(lui)]
        pos = {(u, sample_pos(u), 1) for u in ui_dict.keys()}

        neg = set()
        for u, i, _ in pos:
            for _ in range(self._m):
                neg.add((u, r_int(n_items), 0))

        samples = list(pos)
        samples.extend(list(neg))
        samples = random.sample(samples, len(samples))
        # samples_zip = list(zip(samples))

        for start in range(0, len(samples), batch_size):
            u, i, b = map(np.array, zip(*samples[start:min(start + batch_size, len(samples))]))
            yield u, i, b
