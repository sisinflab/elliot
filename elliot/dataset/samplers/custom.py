from typing import Any, List
import numpy as np
import torch
from scipy.sparse import spmatrix

from elliot.dataset.samplers.base_sampler import PipelineSampler
from elliot.utils.registry import sampler_registry


@sampler_registry.register()
class SparseSampler(PipelineSampler):
    """Replays each row of a sparse matrix (e.g. the ratings matrix), densified on
    demand, one row per sampled index.

    Args:
        sparse (spmatrix): The sparse matrix to sample rows from.
        **params (Any): Forwarded to `PipelineSampler.__init__`.
    """

    def __init__(self, sparse: spmatrix, **params: Any):
        super().__init__(**params)

        self.events = sparse.shape[0]

        # Initializing variables
        self._train = sparse

    def sample(self, it: int) -> np.ndarray:
        """Return the dense row `it` of `self._train` (the training dataloader
        already visits the rows in a random order).

        Args:
            it (int): Event index.

        Returns:
            np.ndarray: The dense row.
        """
        return self._train[it].toarray()

    def collate_fn(self, batch: List[np.ndarray]) -> torch.Tensor:
        """Stack a batch of dense rows into a single tensor.

        Args:
            batch (List[np.ndarray]): The batch of dense rows.

        Returns:
            torch.Tensor: The stacked batch.
        """
        batch = np.vstack(batch)
        return torch.from_numpy(batch)
