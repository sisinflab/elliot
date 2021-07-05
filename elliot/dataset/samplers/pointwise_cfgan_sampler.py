"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np


class Sampler:
    def __init__(self, indexed_ratings, sp_i_train, s_zr, s_pm):
        np.random.seed(42)
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        self._s_zr = s_zr
        self._s_pm = s_pm
        self._sp_i_train = sp_i_train

    def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        s_zr = self._s_zr
        s_pm = self._s_pm
        sp_i_train = self._sp_i_train

        def sample(C_u, N_zr, mask, n):
            u = r_int(n_users)
            ui = ui_dict[u]

            for i in ui:
                mask[n][i] = 1

            for i in range(int(s_zr * n_items)):
                ng = r_int(n_items)
                while ng in ui_dict[u]:
                    ng = r_int(n_items)
                N_zr[n][ng] = 1

            for i in range(int(s_pm * n_items)):
                ng = r_int(n_items)
                while ng in ui_dict[u]:
                    ng = r_int(n_items)
                mask[n][ng] = 1

            C_u[n] = sp_i_train.getrow(u).toarray()

        for batch_start in range(0, events, batch_size):
            C_u, mask, N_zr = np.zeros((batch_size, n_items)), np.zeros((batch_size, n_items)), np.zeros(
                (batch_size, n_items))
            for n, _ in enumerate(range(batch_start, min(batch_start + batch_size, events))):
                sample(C_u, N_zr, mask, n)
            # zip(*[sample(C_u, N_zr, mask, n) for n, _ in enumerate(range(batch_start, min(batch_start + batch_size, events)))])
            yield C_u, mask, N_zr
