"""
Module description:

"""


import torch

from elliot.dataset.samplers.base_sampler import PipelineSampler


class SparseSampler(PipelineSampler):
    def __init__(self, sp_i_train, **params):
        super().__init__(**params)

        self._train = sp_i_train
        self._indices = list(range(self.events))
        self._r_shuffle(self._indices)

    def sample(self, it):
        idx = self._indices[it]
        return self._train[idx].toarray()

    def collate_fn(self, batch):
        return torch.tensor(batch)
