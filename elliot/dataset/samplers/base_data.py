from typing import Iterator, Tuple, Union
import torch
from torch.utils.data import DataLoader

from elliot.dataset.samplers.base_sampler import AbstractSampler

Batch = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


def build_dataloader(
    sampler: AbstractSampler,
    batch_size: int,
    seed: int,
    shuffle: bool = True
) -> DataLoader:
    """Build the training dataloader over `sampler` (already moved to the device
    its batches are built on): a torch `DataLoader` using the sampler both as its
    sampler, yielding a whole batch of event indices at a time (`batch_size=None`
    disables torch's per-sample batching), and as its dataset, building that
    batch at once, in the main process.

    Args:
        sampler (AbstractSampler): The sampler to load from.
        batch_size (int): Batch size.
        seed (int): Seed of the shuffling generator.
        shuffle (bool): Whether to reshuffle every epoch. Defaults to True.

    Returns:
        DataLoader: The configured dataloader.
    """
    sampler.set_batching(batch_size, seed=seed, shuffle=shuffle)
    return DataLoader(sampler, batch_size=None, sampler=sampler, collate_fn=lambda batch: batch)


class CombinedDataLoader:
    """Zips a primary dataloader together with a side-information one, one
    batch's tensors concatenated after the other's, so a model's `train_step`
    sees a single combined batch even though the two were sampled by entirely
    independent samplers (see `AbstractRecommender.side_info_sampler_config`).

    The shorter of the two is cycled - its own iterator restarted (so it keeps
    reshuffling) whenever it runs out - so it never cuts the longer one short;
    when both have the same length neither is cycled.

    Args:
        primary (DataLoader): The main (typically interaction) dataloader;
            sets the combined stream's length whenever it is the longer of the two.
        side_info (DataLoader): The side-information dataloader, zipped alongside.
    """

    def __init__(self, primary: DataLoader, side_info: DataLoader):
        self._primary = primary
        self._side_info = side_info

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        primary_len, side_info_len = len(self._primary), len(self._side_info)

        primary_iter = (
            iter(self._primary) if primary_len >= side_info_len else self._cycle(self._primary)
        )
        side_info_iter = (
            iter(self._side_info) if side_info_len >= primary_len else self._cycle(self._side_info)
        )

        for primary_batch, side_info_batch in zip(primary_iter, side_info_iter):
            yield self._as_tuple(primary_batch) + self._as_tuple(side_info_batch)

    def __len__(self) -> int:
        return max(len(self._primary), len(self._side_info))

    @staticmethod
    def _as_tuple(batch: Batch) -> Tuple[torch.Tensor, ...]:
        # A single-tensor batch is one element, not a sequence of rows
        return (batch,) if isinstance(batch, torch.Tensor) else tuple(batch)

    @staticmethod
    def _cycle(dataloader: DataLoader) -> Iterator[Batch]:
        """Replay `dataloader` indefinitely, restarting its iterator (so it keeps
        reshuffling) every time it is exhausted, rather than caching one pass like
        `itertools.cycle` would.

        Args:
            dataloader (DataLoader): The dataloader to replay indefinitely.

        Returns:
            Iterator[Batch]: An iterator yielding `dataloader`'s batches, forever.
        """
        while True:
            yield from dataloader
