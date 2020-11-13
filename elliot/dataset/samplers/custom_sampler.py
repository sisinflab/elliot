"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np


class Sampler:
    def __init__(self, ratings,
                 random,
                 sample_negative_items_empirically: bool = True
                 ):
        self._ratings = ratings
        self._random = random
        self._sample_negative_items_empirically: bool = sample_negative_items_empirically
        self._users = list(self._ratings.keys())
        self._items = list({k for a in self._ratings.values() for k in a.keys()})

        self.private_users = {p: u for p, u in enumerate(self._users)}
        self.public_users = {v: k for k, v in self.private_users.items()}
        self.private_items = {p: i for p, i in enumerate(self._items)}
        self.public_items = {v: k for k, v in self.private_items.items()}

        self._indexed_ratings = {self.public_users[user]: [self.public_items[i] for i in items.keys()]
                                 for user, items in self._ratings.items()}

    def step(self, events: int, batch_size: int):
        r_int = self._random.randint
        n_users = len(self._users)
        n_items = len(self._items)
        n_batch = events // batch_size
        indexed_ratings = self._indexed_ratings
        user_input, pos_input, neg_input = [], [], []

        for ab in range(n_batch):
            bui, bii, bji = [], [], []
            for cd in range(batch_size):
                u = r_int(n_users)
                ui = set(indexed_ratings[u])
                lui = len(ui)
                if lui == n_items:
                    continue
                i = list(ui)[r_int(lui)]

                j = r_int(n_items)
                while j in ui:
                    j = r_int(n_items)
                bui.append(u)
                bii.append(i)
                bji.append(j)
            user_input.append(np.array(bui)[:, None])
            pos_input.append(np.array(bii)[:, None])
            neg_input.append(np.array(bji)[:, None])
        return user_input, pos_input, neg_input,
