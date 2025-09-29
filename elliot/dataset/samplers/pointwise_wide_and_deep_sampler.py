"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import random
import numpy as np
import itertools

from elliot.dataset.samplers.base_sampler import TraditionalSampler


class Sampler(TraditionalSampler):
    def __init__(self, data, seed=42):
        super().__init__(data.i_train_dict, seed)
        self._sp_i_features = data.sp_i_features
        self._user_encoder = data.user_encoder
        self._item_encoder = data.item_encoder
        """np.random.seed(42)
        random.seed()
        self._data = data
        self._indexed_ratings = self._data.i_train_dict
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(self._indexed_ratings[u])) for u in self._indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}"""

        self._nfeatures = len(data.features)

    """def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict
        n_features = self._nfeatures
        sp_i_features = self._data.sp_i_features
        user_encoder = self._data.user_encoder
        item_encoder = self._data.item_encoder"""

    def _sample(self, **kwargs):
        u = self._r_int(self._nusers)
        u_one_hot = self._user_encoder.transform([[u]])

        ui = self._ui_dict[u]
        lui = self._lui_dict[u]
        if lui == self._nitems:
            self._sample()
        b = random.getrandbits(1)
        if b:
            i = ui[self._r_int(lui)]
        else:
            i = self._r_int(self._nitems)
            while i in ui:
                i = self._r_int(self._nitems)

        i_one_hot = self._item_encoder.transform([[i]])

        f_one_hot = list(itertools.chain.from_iterable([sp_i_feature.getrow(i).toarray()[0].tolist()
                                                        for sp_i_feature in self._sp_i_features]))

        s = []
        s += u_one_hot.toarray()[0].tolist()
        s += i_one_hot.toarray()[0].tolist()
        s += f_one_hot

        return u, i, s, b

        for batch_start in range(0, events, batch_size):
            u, i, s, b = map(np.array,
                             zip(*[sample() for _ in range(batch_start, min(batch_start + batch_size, events))]))
            yield u, i, s, b
