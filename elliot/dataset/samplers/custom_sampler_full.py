"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np

from elliot.dataset.samplers.base_sampler import TraditionalSampler


class Sampler(TraditionalSampler):
    def __init__(self, indexed_ratings, seed=42):
        super().__init__(indexed_ratings, seed)
        self.edge_index = {}
        """np.random.seed(seed)
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

    def step(self, edge_index, events: int, batch_size: int):
        r_int = np.random.randint
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict"""

    def _sample(self, idx, **kwargs):
        ui = self.edge_index[idx]
        u_pos = self._ui_dict[ui[0]]
        lui = self._lui_dict[ui[0]]
        if lui == self._nitems:
            self._sample(idx)
        i = ui[1]

        j = self._r_int(self._nitems)
        while j in u_pos:
            j = self._r_int(self._nitems)
        return ui[0], i, j

        for batch_start in range(0, events, batch_size):
            bui, bii, bij = map(np.array, zip(*[sample(i) for i in range(batch_start, min(batch_start + batch_size, events))]))
            yield bui[:, None], bii[:, None], bij[:, None]
