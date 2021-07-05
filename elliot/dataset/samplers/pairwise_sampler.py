"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np


class Sampler:
    def __init__(self,
                 ratings,
                 users,
                 items
                 ):
        np.random.seed(42)
        self._ratings = ratings
        self._users = users
        self._items = items

    def step(self, events: int):
        r_int = np.random.randint
        n_users = len(self._users)
        n_items = len(self._items)
        users = self._users
        items = self._items
        ratings = self._ratings

        for _ in range(events):
            u = users[r_int(n_users)]
            ui = set(ratings[u].keys())
            lui = len(ui)
            if lui == n_items:
                continue
            i = list(ui)[r_int(lui)]

            v = items[r_int(n_items)]
            while v in ui:
                v = items[r_int(n_items)]

            yield u, i, v
