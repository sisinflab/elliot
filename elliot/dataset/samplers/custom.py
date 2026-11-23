from typing import Any
import numpy as np
import torch
from scipy.sparse import spmatrix

from elliot.dataset.samplers.base_sampler import InteractionSampler
from elliot.utils.registry import sampler_registry


@sampler_registry.register()
class SparseSampler(InteractionSampler):
    """Replays each row of a sparse matrix (e.g. the ratings matrix), densified on
    demand, one row per sampled index.

    Args:
        sparse (spmatrix): The sparse matrix to sample rows from.
        **params (Any): Forwarded to `InteractionSampler.__init__`.
    """

    def __init__(self, sparse: spmatrix, **params: Any):
        super().__init__(**params)

        self.events = sparse.shape[0]

        # Initializing variables: a CSR copy with summed duplicates, as `toarray` would
        train = sparse.tocsr(copy=True)
        train.sum_duplicates()
        self._n_cols = train.shape[1]

        self._register_state("_row_indptr", train.indptr.astype(np.int64))
        self._register_state("_row_indices", train.indices.astype(np.int64))
        self._register_state("_row_data", train.data)

    def sample(self, indices: torch.Tensor) -> torch.Tensor:
        """Return the dense rows `indices` of the matrix (the training dataloader
        already visits the rows in a random order).

        Args:
            indices (torch.Tensor): Row indices.

        Returns:
            torch.Tensor: The dense rows, shape `(batch, n_cols)`, with the matrix's dtype.
        """
        starts = self._row_indptr[indices]
        counts = self._row_indptr[indices + 1] - starts

        # Every non-zero entry of the requested rows: its batch row and its CSR position
        batch_rows = torch.repeat_interleave(torch.arange(len(indices), device=indices.device), counts)
        first_entry = torch.repeat_interleave(starts - (torch.cumsum(counts, 0) - counts), counts)
        positions = first_entry + torch.arange(len(batch_rows), device=indices.device)

        dense = torch.zeros((len(indices), self._n_cols), dtype=self._row_data.dtype, device=indices.device)
        dense[batch_rows, self._row_indices[positions]] = self._row_data[positions]
        return dense
