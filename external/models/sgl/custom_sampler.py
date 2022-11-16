"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from operator import itemgetter
from itertools import chain
import torch
from torch_sparse import SparseTensor


class Sampler:
    def __init__(self, indexed_ratings, iu_dict, all_user_item, n_layers, seed=42):
        np.random.seed(seed)
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._iu_dict = iu_dict
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        self._n_layers = n_layers
        self._all_user_item = all_user_item

    def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict

        def sample():
            u = r_int(n_users)
            ui = ui_dict[u]
            lui = lui_dict[u]
            if lui == n_items:
                sample()
            i = ui[r_int(lui)]

            j = r_int(n_items)
            while j in ui:
                j = r_int(n_items)
            return u, i, j

        for batch_start in range(0, events, batch_size):
            bui, bii, bij = map(np.array,
                                zip(*[sample() for _ in
                                      range(batch_start, min(batch_start + batch_size, events))]))
            sampled_users = set(bui.tolist())
            sampled_items = set(bii.tolist() + bij.tolist())
            for n in range(self._n_layers):
                neighbor_users = set(chain.from_iterable(itemgetter(*sampled_users)(self._ui_dict)))
                sampled_items.update(neighbor_users)
                neighbor_items = set(chain.from_iterable(itemgetter(*sampled_items)(self._iu_dict)))
                sampled_users.update(neighbor_items)
            sampled_items = set(si + self._nusers for si in sampled_items)
            mask_user = np.isin(self._all_user_item[0], list(sampled_users))
            mask_item = np.isin(self._all_user_item[1], list(sampled_items))
            edge_index = self._all_user_item[:, mask_user & mask_item]
            edge_index = torch.tensor(edge_index, dtype=torch.int64)
            adj = SparseTensor(row=torch.cat([edge_index[0], edge_index[1]], dim=0),
                               col=torch.cat([edge_index[1], edge_index[0]], dim=0),
                               sparse_sizes=(self._nusers + self._nitems,
                                             self._nusers + self._nitems))

            yield bui[:, None], bii[:, None], bij[:, None], adj
