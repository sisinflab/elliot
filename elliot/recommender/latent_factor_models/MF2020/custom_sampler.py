"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import random


class Sampler:
    def __init__(self, indexed_ratings, m, sparse_matrix, seed):
        np.random.seed(seed)
        random.seed(seed)
        self._sparse = sparse_matrix
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        self._m = m

    def step(self, batch_size):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict

        # def sample_pos(u):
        #     ui = ui_dict[u]
        #     lui = lui_dict[u]
        #     if lui == n_items:
        #         return None
        #     return ui[r_int(lui)]
        # pos = {(u, i, 1) for u, items in ui_dict.items() for i in items}
        nonzero = self._sparse.nonzero()
        pos = list(zip(*nonzero,np.ones(len(nonzero[0]), dtype=np.int32)))

        neg = list()
        for u, i, _ in pos:
            neg_samples = random.sample(range(n_items), self._m)
            neg += list(zip(np.ones(len(neg_samples), dtype=np.int32) * u, neg_samples, np.zeros(len(neg_samples), dtype=np.int32)))
            pass
            # for _ in range(self._m):
            #     neg.add((u, r_int(n_items), 0))

        # samples = list(pos)
        samples = pos + neg
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

        # for sample in zip(samples):
        #     yield

        for start in range(0, len(samples), batch_size):
            # u, i, b = samples[start:min(start + batch_size, len(samples))]
            yield samples[start:min(start + batch_size, len(samples))]
