"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import random
from operator import itemgetter


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
        self._probs = {u: self.get_probs(v) for u, v in self._ui_dict.items()}
        
    def get_probs(self, ui):
        probs = np.ones(self._nitems)
        probs[ui] = 0
        probs /= np.sum(probs)
        return probs

    def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict
        probs = self._probs

        def sample(bs):
            users = random.sample(self._users, bs)
            pos_items = []
            all_probs = list(itemgetter(*users)(probs))
            for u in users:
                ui = ui_dict[u]
                lui = lui_dict[u]
                if lui == n_items:
                    sample(bs)
                i = ui[r_int(lui)]

                pos_items.append(i)
            return users, pos_items, all_probs

        for batch_start in range(0, events, batch_size):
            batch_stop = min(batch_start + batch_size, events)
            current_batch_size = batch_stop - batch_start
            bui, bii, ps = sample(current_batch_size)
            batch = np.array([bui, bii])
            yield batch, ps
