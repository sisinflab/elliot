import numpy as np

from elliot.dataset.samplers.base_sampler import TraditionalSampler


class Sampler(TraditionalSampler):
    def __init__(self, ui_dict, edge_index, seed=42):
        super().__init__(seed, ui_dict)
        """self._ui_dict = ui_dict
        self._users = list(self._ui_dict.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._ui_dict.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}"""
        self._edge_index = edge_index

    def _sample(self, idx, **kwargs):
        ui = self._edge_index[idx]
        lui = self._lui_dict[ui[0]]
        if lui == self._nitems:
            self._sample(idx)
        i = ui[1]

        return ui[0], i, ui[2] - 1.0

    def prepare_output(self, user, item, r):
        return user, item, r.astype('int64')

    """def step(self, edge_index, events: int, batch_size: int):
        n_items = self._nitems
        lui_dict = self._lui_dict
        edge_index = edge_index.astype(np.int)

        def sample(idx):
            ui = edge_index[idx]
            lui = lui_dict[ui[0]]
            if lui == n_items:
                sample(idx)
            i = ui[1]

            return ui[0], i, ui[2] - 1.0

        for batch_start in range(0, events, batch_size):
            user, item, r = map(np.array, zip(*[sample(i) for i in range(batch_start, min(batch_start + batch_size,
                                                                                          events))]))
            yield user, item, r.astype('int64')"""
