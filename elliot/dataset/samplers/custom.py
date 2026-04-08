"""
Module description:

"""
import numpy as np
import torch

from elliot.dataset.samplers.base_sampler import PipelineSampler
from elliot.utils.registry import sampler_registry


@sampler_registry.register()
class SparseSampler(PipelineSampler):
    def __init__(self, sparse, **params):
        super().__init__(**params)

        self.events = sparse.shape[0]

        self._train = sparse
        self._indices = list(range(self.events))
        self._r_shuffle(self._indices)

    def sample(self, it):
        idx = self._indices[it]
        return self._train[idx].toarray()

    def collate_fn(self, batch):
        batch = np.vstack(batch)
        return torch.from_numpy(batch)
