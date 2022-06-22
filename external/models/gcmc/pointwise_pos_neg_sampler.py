import numpy as np


class Sampler:
    def __init__(self, ui_dict):
        self._ui_dict = ui_dict
        self._users = list(self._ui_dict.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._ui_dict.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

    def step(self, edge_index, events: int, batch_size: int):
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
            yield user, item, r.astype('int64')
