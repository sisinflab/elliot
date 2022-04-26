"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np


class Sampler:
    def __init__(self, indexed_ratings, public_users, public_items, users_tokens, items_tokens, seed=42):
        np.random.seed(seed)
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        self._public_users = public_users
        self._public_items = public_items
        self._users_tokens = {self._public_users[u]: v for u, v in users_tokens.items()}
        self._items_tokens = {self._public_items[i]: v for i, v in items_tokens.items()}

    def step(self, edge_index, events: int, batch_size: int):
        r_int = np.random.randint
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict
        users_tokens = self._users_tokens
        items_tokens = self._items_tokens

        def sample(idx):
            ui = edge_index[idx]
            u_pos = ui_dict[ui[0]]
            lui = lui_dict[ui[0]]
            if lui == n_items:
                sample(idx)
            i = ui[1]

            j = r_int(n_items)
            while j in u_pos:
                j = r_int(n_items)
            return ui[0], i, j, users_tokens[ui[0]], items_tokens[i], items_tokens[j]

        for batch_start in range(0, events, batch_size):
            bui, bii, bij, u_t, i_t_p, i_t_n = map(np.array, zip(*[sample(i) for i in range(batch_start, min(batch_start + batch_size, events))]))
            yield bui[:, None], bii[:, None], bij[:, None], u_t, i_t_p, i_t_n

    @property
    def users_tokens(self):
        return self._users_tokens

    @property
    def items_tokens(self):
        return self._items_tokens
