"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Alberto Carlo Maria Mancino'
__email__ = 'alberto.mancino@poliba.it'

import numpy as np
import random


class Sampler:
    def __init__(self, indexed_ratings, m, transactions, seed=42):
        np.random.seed(seed)
        random.seed(seed)

        self._users = list(indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in indexed_ratings.values() for k in a.keys()})
        self._m = m

        self._nusers = len(self._users)
        self._items_s = set(self._items)
        self._transactions = transactions
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        self._uj_dict = {u: list(set.difference(self._items_s, r)) for u, r in self._ui_dict.items()}
        self._luj_dict = {u: len(v) for u, v in self._uj_dict.items()}

    def step(self, batch_size):

        r_int = np.random.randint
        num_negatives = self._m
        num_positives = 1
        transactions = self._transactions

        user, item, rating = [], [], []

        for _ in range(0, transactions, num_positives + num_negatives):

            u = r_int(self._nusers)
            i = self._ui_dict[u][r_int(self._lui_dict[u])]

            user.append(u)
            item.append(i)
            rating.append(1)

            for _ in range(num_negatives):
                j = self._uj_dict[u][r_int(self._luj_dict[u])]
                user.append(u)
                item.append(j)
                rating.append(0)

        samples_indices = list(range(len(user)))
        random.shuffle(samples_indices)

        user = np.array(user)
        item = np.array(item)
        rating = np.array(rating)

        for start in range(0, len(user), batch_size):
            indeces = samples_indices[start:min(start + batch_size, len(user))]
            yield user[indeces], item[indeces], rating[indeces]
