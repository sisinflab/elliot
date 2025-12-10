"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import random
import numpy as np
from tqdm import tqdm

from elliot.dataset.samplers.base_sampler import AbstractSampler


class PWPosNegSampler(AbstractSampler):
    def __init__(self, **params):
        super().__init__(**params)
        """np.random.seed(42)
        random.seed(42)
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
        lui_dict = self._lui_dict"""

    def _sample(self, **kwargs):
        users = self._r_int(0, self._nusers, size=self.events)
        labels = self._r_int(0, 2, size=self.events)

        items = np.empty(self.events, dtype=np.int64)

        iter_data = tqdm(
            enumerate(zip(users, labels)),
            total=self.events,
            desc="Sampling",
            leave=False
        )

        for idx, (u, b) in iter_data:
            if self._lui_dict[u] == self._nitems:
                while u in users:
                    u = self._r_int(self._nusers)

            ui = self._ui_dict[u]
            lui = self._lui_dict[u]

            if b == 1:
                items[idx] = ui[self._r_int(0, lui)]
            else:
                while True:
                    neg_i = self._r_int(0, self._nitems)
                    if neg_i not in ui:
                        items[idx] = neg_i
                        break

        return users, items, labels
        # u = self._r_int(self._nusers)
        # ui = self._ui_dict[u]
        # lui = self._lui_dict[u]
        # while True:
        #     if lui == self._nitems:
        #         continue
        #     b = random.getrandbits(1)
        #     if b:
        #         i = ui[self._r_int(lui)]
        #     else:
        #         i = self._r_int(self._nitems)
        #         while i in ui:
        #             i = self._r_int(self._nitems)
        #     return u, i, b

        # for batch_start in range(0, events, batch_size):
        #     u, i, b = map(np.array, zip(*[sample() for _ in range(batch_start, min(batch_start + batch_size, events))]))
        #     yield u, i, b
