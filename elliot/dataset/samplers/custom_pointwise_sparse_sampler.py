"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from tqdm import tqdm

from elliot.dataset.samplers.base_sampler import AbstractSampler


class CustomPWSparseSampler(AbstractSampler):
    def __init__(self, **params):
        super().__init__(**params)
        # self._sp_i_train = data.sp_i_train
        """np.random.seed(42)
        self._indexed_ratings = indexed_ratings
        self._sp_i_train = sp_i_train
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}"""

    """def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict"""

    def _sample(self, **kwargs):
        users = self._r_int(0, self._nusers, size=self.events)
        items = np.empty(self.events, dtype=np.int64)
        labels = np.empty(self.events, dtype=np.int64)

        iter_data = tqdm(
            enumerate(users),
            total=self.events,
            desc="Sampling",
            leave=False
        )

        for idx, u in iter_data:
            if self._lui_dict[u] == self._nitems:
                while u in users:
                    u = self._r_int(self._nusers)

            ui = self._ui_dict[u]
            lui = self._lui_dict[u]
            i = ui[self._r_int(lui)]

            items[idx] = i
            labels[idx] = self._indexed_ratings[u][i]

        return users, items, labels
        # u = self._r_int(self._nusers)
        # ui = self._ui_dict[u]
        # lui = self._lui_dict[u]
        # if lui == self._nitems:
        #     return self._sample()
        # i = ui[self._r_int(lui)]
        # r = self._indexed_ratings[u][i]
        # return u, i, r#, self._sp_i_train[u].toarray()[0]
        #
        # """for batch_start in range(0, events, batch_size):
        #     u, i, r, pos = map(np.array, zip(*[sample() for _ in range(batch_start, min(batch_start + batch_size, events))]))
        #     yield u, i, r, pos"""
