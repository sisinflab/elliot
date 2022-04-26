import random
import numpy as np


class Sampler:
    def __init__(self, ui_dict, public_users, public_items, users_tokens, items_tokens, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self._ui_dict = ui_dict
        self._users = list(self._ui_dict.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._ui_dict.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(ui_dict[u])) for u in ui_dict}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        self._public_users = public_users
        self._public_items = public_items
        self._users_tokens = {self._public_users[u]: v for u, v in users_tokens.items()}
        self._items_tokens = {self._public_items[i]: v for i, v in items_tokens.items()}

    def step(self, edge_index, events: int, batch_size: int):
        n_items = self._nitems
        lui_dict = self._lui_dict
        users_tokens = self._users_tokens
        items_tokens = self._items_tokens
        edge_index = edge_index.astype(np.int)

        def sample(idx):
            ui = edge_index[idx]
            lui = lui_dict[ui[0]]
            if lui == n_items:
                sample(idx)
            i = ui[1]

            u_review_tokens = users_tokens[ui[0]]
            i_review_tokens = items_tokens[i]

            return ui[0], i, ui[2], u_review_tokens, i_review_tokens

        for batch_start in range(0, events, batch_size):
            user, item, bit, u_t, i_t = map(np.array, zip(*[sample(i) for i in range(batch_start, min(batch_start + batch_size, events))]))
            yield user, item, bit.astype('float32'), u_t, i_t

    @property
    def users_tokens(self):
        return self._users_tokens

    @property
    def items_tokens(self):
        return self._items_tokens
