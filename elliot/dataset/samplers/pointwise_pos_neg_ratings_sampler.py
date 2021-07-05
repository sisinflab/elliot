"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np


class Sampler:
    def __init__(self, indexed_ratings, sparse_i_ratings):
        np.random.seed(42)
        self._sparse_i_ratings = sparse_i_ratings
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

    def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict
        boolean_list = [0, 1]

        def sample():
            u = r_int(n_users)
            ui = ui_dict[u]
            lui = lui_dict[u]
            if lui == n_items:
                sample()
            np.random.shuffle(boolean_list)

            if boolean_list[0]:
                i = ui[r_int(lui)]
                r = self._indexed_ratings[u][i]
            else:
                i = r_int(n_items)
                while i in ui:
                    i = r_int(n_items)
                r = 0
            return u, i, r

        for batch_start in range(0, events, batch_size):
            u, i, r = map(np.array, zip(*[sample() for _ in range(batch_start, min(batch_start + batch_size, events))]))
            yield u, i, r
