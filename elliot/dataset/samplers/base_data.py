from typing import Any, Iterator, Tuple
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

from elliot.dataset.samplers.base_sampler import AbstractSampler
from elliot.utils import logging as elog
from elliot.utils.enums import SamplerMaterialization
from elliot.utils.utils import get_device, get_num_workers


class PipelineDataset(Dataset):
    """Lazy `Dataset` for pipeline samplers: one `sampler.sample` call per event.

    Args:
        sampler (AbstractSampler): The pipeline sampler to draw samples from.
    """

    def __init__(self, sampler: AbstractSampler):
        super().__init__()
        self.sampler = sampler

    def __len__(self) -> int:
        return self.sampler.events

    def __getitem__(self, index: int) -> Any:
        return self.sampler.sample(index)


def build_dataset(sampler: AbstractSampler) -> Dataset:
    """Wrap `sampler` into the `torch.utils.data.Dataset` matching its declared
    `SamplerType`: an eagerly materialized `TensorDataset` for `TRADITIONAL`, or a
    lazy `PipelineDataset`/`SequentialDataset` for `PIPELINE`/`SEQUENTIAL`. Any
    `collate_fn` the sampler itself defines is attached to the returned dataset.

    Args:
        sampler (AbstractSampler): The sampler to wrap.

    Returns:
        Dataset: The dataset built from `sampler`.

    Raises:
        ValueError: If `sampler.type` is not a recognized `SamplerType`.
    """
    match sampler.materialization:
        # Eagerly materialize the whole stream into one in-memory tensor dataset
        case SamplerMaterialization.TRADITIONAL:
            samples = sampler.sample_full()
            columns = (torch.tensor(column) for column in zip(*samples))
            # Integer columns become int64, a float one (e.g. ratings) is not truncated
            tensors = tuple(t if t.is_floating_point() else t.long() for t in columns)
            dataset = TensorDataset(*tensors)

        # Lazy: sample one event per __getitem__ call
        case SamplerMaterialization.PIPELINE:
            dataset = PipelineDataset(sampler)

        case _:
            raise ValueError(f"Invalid sampler type {sampler.type}")

    # Forward the sampler's own collate_fn, if it declares one
    collate_fn = getattr(sampler, 'collate_fn', None)
    if collate_fn is not None:
        setattr(dataset, 'collate_fn', collate_fn)

    return dataset


def _seed_worker(worker_id: int) -> None:
    """Reseed the RNG of the sampler a dataloader worker draws from.

    Every worker receives an identical copy of the parent's sampler, own RNG state
    included, so without this every worker would draw the same samples. PyTorch
    gives every worker its own torch seed (`base_seed + worker_id`), which reseeds
    the worker's copy of the sampler.

    Args:
        worker_id (int): The worker's index (unused; already folded into the torch seed).
    """
    worker_info = torch.utils.data.get_worker_info()
    sampler = getattr(worker_info.dataset, "sampler", None)
    if isinstance(sampler, AbstractSampler):
        sampler.reseed(torch.initial_seed())


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    shuffle: bool = True
) -> DataLoader:
    """Build the training `DataLoader` over `dataset`, forwarding any `collate_fn`
    attached to it by `build_dataset`.

    Worker processes (see `ExperimentConfig.num_workers`) are only used for lazy
    `PipelineDataset`s, where each sample is built on demand: an eagerly materialized
    `TensorDataset` is just indexed, so workers would only add overhead. With
    workers, shuffling draws from a dedicated generator seeded with `seed` (also the
    source of every worker's seed) and workers are kept alive across epochs; with
    none, the loader is built exactly as before (shuffling from torch's global RNG).

    Args:
        dataset (Dataset): The dataset to load from, as returned by `build_dataset`.
        batch_size (int): Batch size.
        seed (int): Seed for the shuffling generator and worker seeds, used only
            when worker processes are enabled.
        shuffle (bool): Whether to reshuffle every epoch. Defaults to True.

    Returns:
        DataLoader: The configured dataloader.
    """
    num_workers = get_num_workers() if isinstance(dataset, PipelineDataset) else 0

    worker_kwargs = {}
    if num_workers > 0:
        elog.get_logger("DataLoader").info(
            "Starting dataloader workers", extra={"context": {"num_workers": num_workers}}
        )
        worker_kwargs = dict(
            num_workers=num_workers,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(seed),
            persistent_workers=True,
            pin_memory=get_device().type == "cuda",
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=getattr(dataset, 'collate_fn', None),
        shuffle=shuffle,
        **worker_kwargs
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
        primary (DataLoader): The main (typically interaction) dataloader; sets
            the combined stream's length whenever it is the longer of the two.
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
            yield tuple(primary_batch) + tuple(side_info_batch)

    def __len__(self) -> int:
        return max(len(self._primary), len(self._side_info))

    @staticmethod
    def _cycle(dataloader: DataLoader) -> Iterator:
        """Replay `dataloader` indefinitely, restarting its iterator (so it keeps
        reshuffling) every time it is exhausted, rather than caching one pass like
        `itertools.cycle` would.

        Args:
            dataloader (DataLoader): The dataloader to replay indefinitely.

        Returns:
            Iterator: An iterator yielding `dataloader`'s batches, forever.
        """
        while True:
            yield from dataloader
