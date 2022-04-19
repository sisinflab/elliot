import numpy as np


class Sampler:
    def __init__(self, ui_dict, public_users, public_items, users_tokens, items_tokens, pos_user, pos_item):
        self._ui_dict = ui_dict
        self._users = list(self._ui_dict.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._ui_dict.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(ui_dict[u])) for u in ui_dict}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        last_private_user = list(public_users.keys())[-1] + 1
        last_public_user = len(public_users)
        last_private_item = list(public_items.keys())[-1] + 1
        last_publict_item = len(public_items)
        self._public_users = public_users
        self._public_users[last_private_user] = last_public_user
        self._public_items = public_items
        self._public_items[last_private_item] = last_publict_item
        self._users_tokens = {self._public_users[u]: v for u, v in users_tokens.items()}
        self._items_tokens = {self._public_items[i]: v for i, v in items_tokens.items()}
        self._pos_user = pos_user
        self._pos_item = pos_item

    def step(self, edge_index, events: int, batch_size: int):
        n_items = self._nitems
        lui_dict = self._lui_dict
        users_tokens = self._users_tokens
        items_tokens = self._items_tokens
        pos_user = self._pos_user
        pos_item = self._pos_item
        edge_index = edge_index.astype(np.int)

        def sample(idx):
            ui = edge_index[idx]
            lui = lui_dict[ui[0]]
            if lui == n_items:
                sample(idx)
            i = ui[1]

            u_review_tokens = users_tokens[ui[0]]
            i_review_tokens = items_tokens[i]
            u_pos_user = pos_user[ui[0]]
            i_pos_item = pos_item[i]

            return ui[0], i, ui[2], u_review_tokens, i_review_tokens, u_pos_user, i_pos_item

        for batch_start in range(0, events, batch_size):
            user, item, bit, u_t, i_t, u_p, i_p = map(np.array, zip(*[sample(i) for i in range(batch_start, min(batch_start + batch_size, events))]))
            yield user, item, bit.astype('float32'), u_t, i_t, u_p, i_p

    @property
    def users_tokens(self):
        return self._users_tokens

    @property
    def items_tokens(self):
        return self._items_tokens

    @property
    def pos_users(self):
        return self._pos_user

    @property
    def pos_items(self):
        return self._pos_item


