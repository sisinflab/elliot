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

    def step(self):
        r_int = np.random.randint
        n_users = self._nusers
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

        # def sample():
        #     u = r_int(n_users)
        #     ui = ui_dict[u]
        #     lui = lui_dict[u]
        #     if lui == n_items:
        #         sample()
        #     i = ui[r_int(lui)]
        #
        #     j = r_int(n_items)
        #     while j in ui:
        #         j = r_int(n_items)
        #     return u, i, j

        for sample in zip(samples):
            yield sample
