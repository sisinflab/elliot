"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import random


class Sampler:
    def __init__(self, indexed_ratings, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

    def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict

        def sample(bs):
            users = random.sample(self._users, bs)
            pos_items, neg_items = [], []
            for u in users:
                ui = ui_dict[u]
                lui = lui_dict[u]
                if lui == n_items:
                    sample(bs)
                i = ui[r_int(lui)]

                j = r_int(n_items)
                while j in ui:
                    j = r_int(n_items)
                pos_items.append(i), neg_items.append(j)
            return users, pos_items, neg_items

        for batch_start in range(0, events, batch_size):
            batch_stop = min(batch_start + batch_size, events)
            current_batch_size = batch_stop - batch_start
            bui, bii, bij = sample(current_batch_size)
            bui, bii, bij = np.array(bui), np.array(bii), np.array(bij)
            yield bui[:, None], bii[:, None], bij[:, None]
