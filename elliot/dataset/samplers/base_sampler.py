from typing import Any, Dict, List, Optional, Tuple, Union
import random
import time
import numpy as np
import torch
from abc import ABC, abstractmethod
from functools import partial
from logging import LoggerAdapter
from scipy.sparse import csr_matrix
from tqdm import tqdm
from torch.utils.data import Dataset, TensorDataset

from elliot.utils import logging as elog
from elliot.utils.enums import SamplerType, SessionStrategy


class AbstractSampler(ABC):
    """Base class every sampler registered in `sampler_registry` implements.

    Subclasses must implement `sample(it)`; `sample_full()`/`sample_eval(it)` are
    optional hooks, no-ops by default.

    Args:
        users (List[int]): Private user indices in this split's domain.
        items (List[int]): Private item indices in this split's domain.
        n_users (int): Total number of users.
        n_items (int): Total number of items.
        seed (int): Random seed for reproducibility.
        logger (LoggerAdapter, optional): Logging instance. Defaults to None, building
            a fresh one via `elliot.utils.logging.get_logger`.
        **kwargs (Any): Unused; absorbs any extra keyword arguments forwarded by
            `sampler_registry.get()` that a concrete sampler doesn't itself declare.
    """

    type: SamplerType

    def __init__(
        self,
        users: List[int],
        items: List[int],
        n_users: int,
        n_items: int,
        seed: int,
        logger: Optional[LoggerAdapter] = None,
        **kwargs: Any
    ):
        self.logger = logger or elog.get_logger(self.__class__.__name__, seed=seed)

        # Initializing variables
        self._users = users
        self._nusers = n_users
        self._items = items
        self._nitems = n_items

        np.random.seed(seed)
        random.seed(seed)

        # Cache bound RNG methods so subclasses don't re-derive them each call
        self._r_int = partial(np.random.randint)
        self._r_choice = partial(np.random.choice)
        self._r_shuffle = partial(random.shuffle)
        self._r_sample = partial(random.sample)

        self.events: int = 0

    def sample_full(self):
        """Optional hook: build the whole sample stream at once (used by
        `SamplerType.TRADITIONAL` samplers via `build_dataset`). Default is a no-op;
        see `TraditionalSampler.sample_full` for the actual implementation.
        """
        pass

    @abstractmethod
    def sample(self, it: int) -> Any:
        """Build a single sample for event index `it`.

        Args:
            it (int): Event index.

        Returns:
            Any: The sample, whose shape is defined by the concrete sampler.
        """
        raise NotImplementedError()

    def sample_eval(self, it: int) -> Any:
        """Optional hook: `sample`'s evaluation-time counterpart (used by
        `TraditionalSampler.sample_full` when `val=True`). Default is a no-op.

        Args:
            it (int): Event index.

        Returns:
            Any: The sample, whose shape is defined by the concrete sampler.
        """
        pass

    def read_features(self, *args):
        return args

    def read_features_eval(self, *args):
        return args


class TraditionalSampler(AbstractSampler):
    """Materializes its whole event stream at once, via `sample_full()`, for
    consumption as a single in-memory `TensorDataset` (see `build_dataset`,
    `SamplerType.TRADITIONAL`).

    Args:
        train_dict (Dict[int, Dict[int, float]]): Private-id-keyed ratings dict for
            this split (`user -> {item: rating}`).
        transactions (int): Number of events to sample.
        **kwargs (Any): Forwarded to `AbstractSampler.__init__`.
    """

    type = SamplerType.TRADITIONAL

    def __init__(
        self,
        train_dict: Dict[int, Dict[int, float]],
        transactions: int,
        **kwargs: Any
    ):
        super().__init__(**kwargs)

        # Initializing variables
        self.events = transactions
        self._indexed_ratings = train_dict

        # Per-user item list and its length, cached for O(1) sampling
        self._ui_dict = {u: list(set(self._indexed_ratings[u])) for u in self._indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

    def sample_full(self, val: bool = False) -> List[Any]:
        """Build every sample in the event stream at once, shuffled.

        Args:
            val (bool): If True, use `sample_eval`/`read_features_eval` instead of
                `sample`/`read_features`. Defaults to False.

        Returns:
            List[Any]: The full, shuffled sample stream.
        """
        start = time.time()

        iter_data = tqdm(
            range(self.events),
            total=self.events,
            desc="Sampling",
            leave=False
        )
        samples = []

        # Pick sample()/read_features() or their eval-time counterparts
        sample_fn = self.sample if not val else self.sample_eval
        read_features_fn = self.read_features if not val else self.read_features_eval

        for it in iter_data:
            output = sample_fn(it)
            output = read_features_fn(*output)

            # A hook may explode one sample into several (e.g. windowed sequences)
            if isinstance(output, list):
                samples.extend(output)
            else:
                samples.append(output)

        self._r_shuffle(samples)

        end = time.time()

        self.logger.debug(
            "Completed sampling",
            extra={"context": {"duration_sec": round(end - start, 4), "events": len(samples)}}
        )

        return samples


class PipelineSampler(AbstractSampler):
    """Lazily sampled counterpart to `TraditionalSampler`: `sample(it)` is called
    on demand, once per dataset index, by `PipelineDataset` (see `build_dataset`,
    `SamplerType.PIPELINE`).

    Args:
        train_dict (Dict[int, Dict[int, float]]): Private-id-keyed ratings dict for
            this split (`user -> {item: rating}`).
        transactions (int): Number of events to sample.
        **kwargs (Any): Forwarded to `AbstractSampler.__init__`.
    """

    type = SamplerType.PIPELINE

    def __init__(
        self,
        train_dict: Dict[int, Dict[int, float]],
        transactions: int,
        **kwargs: Any
    ):
        super().__init__(**kwargs)

        # Initializing variables
        self.events = transactions
        self._indexed_ratings = train_dict

        # Per-user item list and its length, cached for O(1) sampling
        self._ui_dict = {u: list(set(self._indexed_ratings[u])) for u in self._indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

    def collate_fn(self, batch: List[Any]) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Shuffle a batch of samples and stack each tuple position into its own tensor.

        Args:
            batch (List[Any]): The batch of samples, each a tuple of same-length values.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]: One tensor per tuple position.
        """
        self._r_shuffle(batch)

        # Transpose the batch of tuples into one tensor per tuple position
        tensors = tuple(
            torch.tensor(x, dtype=torch.long) for x in zip(*batch)
        )

        return tensors


class SessionSampler(AbstractSampler):
    """Base class for samplers operating on `Sessions`' flat item tape.

    Subclasses receive the same globally sorted (by user, session, timestamp)
    tape regardless of strategy: the only thing that changes between FLAT and
    SESSION_ONLY is which boundary array bounds a sequence (per-user vs.
    per-session), resolved once here so subclasses never branch on strategy.

    Args:
        flat_items (np.ndarray): Item id per flat tape position.
        flat_users (np.ndarray): Owning (private) user index per flat tape position.
        flat_session (np.ndarray): Owning (private) session index per flat tape position.
        user_offsets (np.ndarray): Per-user boundary array over the flat tape.
        session_offsets (np.ndarray): Per-session boundary array over the flat tape.
        sparse (csr_matrix): Train interaction matrix, used to exclude seen items when
            sampling negatives.
        strategy (SessionStrategy): FLAT or SESSION_ONLY. Defaults to FLAT.
        max_seq_len (int): Maximum sequence length built from the flat tape. Defaults
            to 50.
        neg_samples (int): Number of negatives sampled per target, or 0 to disable
            negative sampling. Defaults to 0.
        **kwargs (Any): Forwarded to `AbstractSampler.__init__`.
    """

    type = SamplerType.SEQUENTIAL

    def __init__(
        self,
        flat_items: np.ndarray,
        flat_users: np.ndarray,
        flat_session: np.ndarray,
        user_offsets: np.ndarray,
        session_offsets: np.ndarray,
        sparse: csr_matrix,
        strategy: SessionStrategy = SessionStrategy.FLAT,
        max_seq_len: int = 50,
        neg_samples: int = 0,
        **kwargs: Any
    ):
        super().__init__(**kwargs)

        # Initializing variables
        self._flat_items = np.asarray(flat_items)
        self._flat_users = np.asarray(flat_users)
        self._flat_session = np.asarray(flat_session)
        self._user_offsets = np.asarray(user_offsets)
        self._session_offsets = np.asarray(session_offsets)
        self._sparse = sparse

        self.strategy = strategy if isinstance(strategy, SessionStrategy) else SessionStrategy(strategy)
        self.max_seq_len = max_seq_len

        self._neg_samples = neg_samples
        self._niid = self._nitems
        self._padding_token = self._nitems

        # Which boundary array bounds a sequence: per-session (SESSION_ONLY) or per-user (FLAT)
        self._boundaries = (
            self._session_offsets if self.strategy == SessionStrategy.SESSION_ONLY else self._user_offsets
        )
        self._valid_target_indices = self._compute_valid_targets()

        self.events = len(self._valid_target_indices)

    def _compute_valid_targets(self) -> np.ndarray:
        """A flat position is a valid next-item target iff it isn't the first
        position of its boundary segment (it needs at least one predecessor).

        Returns:
            np.ndarray: The valid target flat tape positions.
        """
        n = len(self._flat_items)
        valid_mask = np.ones(n, dtype=bool)

        # Exclude each boundary segment's own first position (it has no predecessor)
        starts = self._boundaries[:-1]
        active_starts = starts[starts < n]
        valid_mask[active_starts] = False

        return np.arange(n)[valid_mask]

    def _boundary_id_of(self, idx: int) -> int:
        """Return the boundary segment (user, if FLAT; session, if SESSION_ONLY)
        owning flat tape position `idx`.

        Args:
            idx (int): Flat tape position.

        Returns:
            int: The owning boundary segment id.
        """
        return (
            int(self._flat_users[idx]) if self.strategy == SessionStrategy.FLAT
            else int(self._flat_session[idx])
        )

    def _boundary_start_of(self, idx: int) -> int:
        """Return the flat tape position where `idx`'s boundary segment starts.

        Args:
            idx (int): Flat tape position.

        Returns:
            int: The start position of the owning boundary segment.
        """
        return int(self._boundaries[self._boundary_id_of(idx)])

    def _build_padded_sequence(self, end_idx: int, boundary_start: int) -> Tuple[torch.Tensor, int]:
        """Build a padded item sequence ending (exclusively) at `end_idx`, clipped to
        `boundary_start` and to `max_seq_len` items.

        Args:
            end_idx (int): Flat tape position, exclusive, ending the sequence.
            boundary_start (int): Flat tape position where the owning boundary
                segment starts; the sequence never reaches further back than this.

        Returns:
            Tuple[torch.Tensor, int]: The padded sequence tensor (length
                `max_seq_len`) and its true (pre-padding) length.
        """
        # Never reach further back than the boundary segment's own start
        start_idx = max(boundary_start, end_idx - self.max_seq_len)
        seq_array = self._flat_items[start_idx:end_idx]
        seq_len = len(seq_array)

        # Left-align real items, pad the rest
        seq_tensor = torch.full((self.max_seq_len,), self._padding_token, dtype=torch.long)
        if seq_len:
            seq_tensor[:seq_len] = torch.from_numpy(seq_array.copy())

        return seq_tensor, seq_len

    def _sample_negatives(self, owner_user: int, k: int, exclude_item: Optional[int] = None) -> List[int]:
        """Uniformly sample `k` negative items for `owner_user`, excluding items seen
        in `self.sparse` and, optionally, `exclude_item`.

        Args:
            owner_user (int): Private user index whose seen items are excluded.
            k (int): Number of negatives to sample.
            exclude_item (int, optional): An extra item id to exclude (e.g. the
                current positive target). Defaults to None.

        Returns:
            List[int]: The sampled negative item ids.
        """
        # This user's seen items, read directly off the CSR index (already sorted)
        u_start = self._sparse.indptr[owner_user]
        u_end = self._sparse.indptr[owner_user + 1]
        seen_items = self._sparse.indices[u_start:u_end]

        negatives = []
        while len(negatives) < k:
            cand = int(self._r_int(self._niid))
            if exclude_item is not None and cand == exclude_item:
                continue

            # Binary search since seen_items is sorted
            pos = np.searchsorted(seen_items, cand)
            if pos < len(seen_items) and seen_items[pos] == cand:
                continue

            negatives.append(cand)

        return negatives


class PipelineDataset(Dataset):
    """Lazy `Dataset` for `SamplerType.PIPELINE` samplers, replaying `sampler.sample`
    `m + 1` times per event (`m` extra negatives sampled per positive, when the
    sampler declares one via its own `m` attribute).

    Args:
        sampler (AbstractSampler): The pipeline sampler to draw samples from.
    """

    def __init__(self, sampler: AbstractSampler):
        super().__init__()
        self.sampler = sampler
        self.m = getattr(sampler, 'm', 0)

    def __len__(self) -> int:
        return self.sampler.events * (self.m + 1)

    def __getitem__(self, index: int) -> Any:
        real_idx = index // (self.m + 1)
        return self.sampler.sample(real_idx)


class SequentialDataset(Dataset):
    """Lazy dataset for samplers whose `sample(it)` already returns a tuple of
    built tensors (e.g. padded sequences). Left to PyTorch's default collate,
    which stacks each tuple position independently.

    Args:
        sampler (AbstractSampler): The sampler to draw samples from.
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
    match sampler.type:
        # Eagerly materialize the whole stream into one in-memory tensor dataset
        case SamplerType.TRADITIONAL:
            samples = sampler.sample_full()
            tensors = tuple(torch.tensor(x, dtype=torch.long) for x in zip(*samples))
            dataset = TensorDataset(*tensors)

        # Lazy: sample one event per __getitem__ call
        case SamplerType.PIPELINE:
            dataset = PipelineDataset(sampler)

        case SamplerType.SEQUENTIAL:
            dataset = SequentialDataset(sampler)

        case _:
            raise ValueError(f"Invalid sampler type {sampler.type}")

    # Forward the sampler's own collate_fn, if it declares one
    collate_fn = getattr(sampler, 'collate_fn', None)
    if collate_fn is not None:
        setattr(dataset, 'collate_fn', collate_fn)

    return dataset
