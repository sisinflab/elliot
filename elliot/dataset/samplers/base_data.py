from typing import Iterator, Optional, Tuple, Union
import math
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from elliot.dataset.samplers.base_sampler import AbstractSampler

Batch = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


class SamplerDataset(Dataset):
    """Torch dataset over a sampler (already moved to the device its batches are
    built on), indexed by a whole batch of event indices at once.

    Args:
        sampler (AbstractSampler): The sampler.
    """

    def __init__(self, sampler: AbstractSampler):
        super().__init__()
        self.sampler = sampler

    def __len__(self) -> int:
        return self.sampler.events

    def __getitem__(self, indices: torch.Tensor) -> Batch:
        """Build the batch for a batch of events.

        Args:
            indices (torch.Tensor): Event indices, on the sampler's device.

        Returns:
            Batch: The batch built by `sampler.sample`.
        """
        return self.sampler.sample(indices)

    @staticmethod
    def collate(batch: Batch) -> Batch:
        """Return the batch as built by the sampler (torch's default conversion
        would turn its tuple into a list).

        Args:
            batch (Batch): The batch.

        Returns:
            Batch: The same batch.
        """
        return batch


class BatchIndexSampler(Sampler):
    """Torch sampler yielding the event indices one whole batch at a time, as
    tensors on the sampler's device: every epoch, a fresh permutation of the
    events (or their order, if not shuffling) split into batches.

    Args:
        dataset (SamplerDataset): The dataset to index.
        batch_size (int): Batch size; the last batch may be smaller.
        seed (int): Seed of the shuffling generator.
        shuffle (bool): Whether to reshuffle every epoch. Defaults to True.
    """

    def __init__(self, dataset: SamplerDataset, batch_size: int, seed: int, shuffle: bool = True):
        super().__init__()
        self.n_events = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.device = dataset.sampler.device
        self._generator = AbstractSampler.make_generator(
            seed, AbstractSampler.SHUFFLE_STREAM, self.device
        )

    def __len__(self) -> int:
        return math.ceil(self.n_events / self.batch_size)

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.shuffle:
            order = torch.randperm(self.n_events, generator=self._generator, device=self.device)
        else:
            order = torch.arange(self.n_events, device=self.device)

        yield from order.split(self.batch_size)


def build_dataset(sampler: AbstractSampler, device: Optional[Union[str, torch.device]] = None) -> SamplerDataset:
    """Wrap `sampler` into the `SamplerDataset` a dataset caches, after moving its
    state to the device its batches are built on.

    Args:
        sampler (AbstractSampler): The sampler to wrap.
        device (Union[str, torch.device], optional): Where the sampler builds its
            batches, i.e. the consuming model's device. Defaults to None, the CPU.

    Returns:
        SamplerDataset: The dataset built from `sampler`.
    """
    return SamplerDataset(sampler.to(device or "cpu"))


def build_dataloader(
    dataset: SamplerDataset,
    batch_size: int,
    seed: int,
    shuffle: bool = True
) -> DataLoader:
    """Build the training dataloader over `dataset`: a torch `DataLoader` whose
    `BatchIndexSampler` hands each batch's indices to the dataset at once
    (`batch_size=None` disables torch's per-sample batching), in the main process.

    Args:
        dataset (SamplerDataset): The dataset to load from, as returned by `build_dataset`.
        batch_size (int): Batch size.
        seed (int): Seed of the shuffling generator.
        shuffle (bool): Whether to reshuffle every epoch. Defaults to True.

    Returns:
        DataLoader: The configured dataloader.
    """
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=BatchIndexSampler(dataset, batch_size=batch_size, seed=seed, shuffle=shuffle),
        collate_fn=SamplerDataset.collate
    )


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
