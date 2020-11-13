"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'


class Sampler:
    def __init__(self, train,
                 random
                 ):
        self._train = train
        self._random = random

    def step(self, users: int, batch_size: int):
        train = self._train
        shuffled_list = self._random.sample(range(users), users)

        for start_idx in range(0, users, batch_size):
            end_idx = min(start_idx + batch_size, users)
            yield train[shuffled_list[start_idx:end_idx]]
