"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np

from elliot.dataset.samplers.base_sampler import TraditionalSampler


class BPRMFSampler(TraditionalSampler):
    def __init__(self, indexed_ratings, seed=42):
        super().__init__(seed, indexed_ratings)
        """np.random.seed(seed)
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}"""
        self._freq_users = np.zeros(self._nusers, dtype=np.int64)
        self._freq_items = np.zeros(self._nitems, dtype=np.int64)
        # self._freq_users = dict.fromkeys(self._users, 0)
        # self._freq_items = dict.fromkeys(self._items, 0)

    @property
    def freq_users(self):
        return dict(enumerate(self._freq_users))

    @property
    def freq_items(self):
        return dict(enumerate(self._freq_items))

    def _sample(self, bsize, **kwargs):
        users = self._r_int(0, self._nusers, size=bsize)
        self._freq_users[users] += 1

        pos_items = np.empty(bsize, dtype=np.int64)
        neg_items = np.empty(bsize, dtype=np.int64)

        for idx, u in enumerate(users):
            if self._lui_dict[u] == self._nitems:
                self._freq_users[u] -= 1
                while u in users:
                    u = self._r_int(self._nusers)
                self._freq_users[u] += 1

            ui = self._ui_dict[u]
            lui = self._lui_dict[u]

            i = ui[self._r_int(lui)]
            pos_items[idx] = i
            self._freq_items[i] += 1

            while True:
                j = self._r_int(self._nitems)
                if j not in ui:
                    neg_items[idx] = j
                    self._freq_items[j] += 1
                    break

        return users[:, None], pos_items[:, None], neg_items[:, None]
        u = self._r_int(self._nusers)
        self._freq_users[u] += 1
        ui = self._ui_dict[u]
        lui = self._lui_dict[u]
        if lui == self._nitems:
            self._freq_users[u] -= 1
            return self._sample()
        i = ui[self._r_int(lui)]
        self._freq_items[i] += 1

        j = self._r_int(self._nitems)
        while j in ui:
            j = self._r_int(self._nitems)
        self._freq_items[j] += 1
        return u, i, j

    # def prepare_output(self, *args):
    #     return (r[:, None] for r in args)

    """def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict

        def sample():
            u = r_int(n_users)
            self._freq_users[u] += 1
            ui = ui_dict[u]
            lui = lui_dict[u]
            if lui == n_items:
                self._freq_users[u] -= 1
                sample()
            i = ui[r_int(lui)]
            self._freq_items[i] += 1

            j = r_int(n_items)
            while j in ui:
                j = r_int(n_items)
            self._freq_items[j] += 1
            return u, i, j

        for batch_start in range(0, events, batch_size):
            bui, bii, bij = map(np.array,
                                zip(*[sample() for _ in
                                      range(batch_start, min(batch_start + batch_size, events))]))
            yield bui[:, None], bii[:, None], bij[:, None]"""
