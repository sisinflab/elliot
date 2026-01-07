"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np

from elliot.dataset.samplers.base_sampler import TraditionalSampler, PipelineSampler


class PairWiseSampler(PipelineSampler):
    def __init__(self, **params):
        super().__init__(**params)

        self._sampled_users = self._sample_users()

        self._freq_users = np.zeros(self._nusers, dtype=np.int64)
        self._freq_items = np.zeros(self._nitems, dtype=np.int64)

    def sample(self, it):
        u = self._sampled_users[it]
        ui = self._ui_dict[u]
        lui = self._lui_dict[u]

        if lui == self._nitems:
            while u in self._sampled_users:
                u = self._r_int(self._nusers)
        self._freq_users[u] += 1

        i = ui[self._r_int(lui)]
        self._freq_items[i] += 1

        j = self._r_int(self._nitems)
        while j in ui:
            j = self._r_int(self._nitems)
        self._freq_items[j] += 1

        return u, i, j

    def _sample_users(self):
        return self._r_int(0, self._nusers, size=self.events)

    @property
    def freq_users(self):
        return dict(enumerate(self._freq_users))

    @property
    def freq_items(self):
        return dict(enumerate(self._freq_items))


class PairWiseBatchSampler(PairWiseSampler):
    def __init__(self, b_size, **params):
        self.b_size = b_size
        super().__init__(**params)

    def _sample_users(self):
        sampled_users = []

        for b_start in range(0, self.events, self.b_size):
            b_stop = min(b_start + self.b_size, self.events)
            current_b_size = b_stop - b_start

            b_users = self._r_sample(self._users, k=current_b_size)

            sampled_users.extend(b_users)

        return sampled_users


class MFPairWiseSampler(PipelineSampler):
    def __init__(self, m, **params):
        super().__init__(**params)

        self._pos = [(u, i) for u, items in self._ui_dict.items() for i in items]
        self.m = m

    def sample(self, it):
        u, i = self._pos[it]
        ui = self._ui_dict[u]

        samples = set()
        for _ in range(self.m):
            j = self._r_int(self._nitems)
            while j in ui:
                j = self._r_int(self._nitems)
            samples.add((u, i, j))

        return list(samples)

    def collate_fn(self, batch):
        concatenated = []
        for lst in batch:
            concatenated.extend(lst)

        return super().collate_fn(concatenated)
