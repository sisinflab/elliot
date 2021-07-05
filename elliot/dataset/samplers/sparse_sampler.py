"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import random


class Sampler:
    def __init__(self, sp_i_train
                 ):
        random.seed(42)
        self._train = sp_i_train

    def step(self, users: int, batch_size: int):
        train = self._train
        shuffled_list = random.sample(range(users), users)

        for start_idx in range(0, users, batch_size):
            end_idx = min(start_idx + batch_size, users)
            yield train[shuffled_list[start_idx:end_idx]].toarray()
