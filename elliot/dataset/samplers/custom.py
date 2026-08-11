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

        # Pre-shuffle row order once; sample() just walks through it
        self._indices = list(range(self.events))
        self._r_shuffle(self._indices)

    def sample(self, it: int) -> np.ndarray:
        """Return the dense row of `self._train` at shuffled position `it`.

        Args:
            it (int): Event index.

        Returns:
            np.ndarray: The dense row.
        """
        idx = self._indices[it]
        return self._train[idx].toarray()

    def collate_fn(self, batch: List[np.ndarray]) -> torch.Tensor:
        """Stack a batch of dense rows into a single tensor.

        Args:
            batch (List[np.ndarray]): The batch of dense rows.

        Returns:
            torch.Tensor: The stacked batch.
        """
        batch = np.vstack(batch)
        return torch.from_numpy(batch)
