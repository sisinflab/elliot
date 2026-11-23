from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from itertools import chain
from logging import LoggerAdapter
import numpy as np
import torch
from scipy.sparse import csr_matrix

from elliot.utils import logging as elog
from elliot.utils.enums import SamplerType, SessionStrategy


class AbstractSampler(ABC):
    """Base class of every sampler.

    A sampler builds a whole batch at once: `sample(indices)` turns a batch of
    event indices into the batch's tensors, with tensor operations on
    `self.device`. Every epoch, `BatchIndexSampler` splits a fresh permutation of
    the `self.events` indices into batches, so every draw changes across epochs.

    Its state is built on CPU at construction time; every tensor `sample` reads is
    registered with `_register_state`, so that `to(device)` can move it.

    Args:
        seed (int): Random seed.
        users (List[int], optional): Private user indices of the split.
        items (List[int], optional): Private item indices of the split.
        n_users (int): Total number of users. Defaults to 0.
        n_items (int): Total number of items. Defaults to 0.
        logger (LoggerAdapter, optional): Logging instance.
        **kwargs (Any): Ignored; absorbs what `sampler_registry.get()` forwards.
    """

    type: SamplerType

    # Independent RNG streams of one seed
    INIT_STREAM = 0     # draws made once, at construction time
    BATCH_STREAM = 1    # draws made by `sample`
    SHUFFLE_STREAM = 2  # epoch permutations

    def __init__(
        self,
        seed: int,
        users: Optional[List[int]] = None,
        items: Optional[List[int]] = None,
        n_users: int = 0,
        n_items: int = 0,
        logger: Optional[LoggerAdapter] = None,
        **kwargs: Any
    ):
        self.logger = logger or elog.get_logger(self.__class__.__name__, seed=seed)

        self._users = users if users is not None else []
        self._nusers = n_users
        self._items = items if items is not None else []
        self._nitems = n_items

        self.events: int = 0

        self._seed = seed
        self.device = torch.device("cpu")
        self._state_names: List[str] = []

        self._init_generator = self.make_generator(seed, self.INIT_STREAM)
        self._generator = self.make_generator(seed, self.BATCH_STREAM)

    @abstractmethod
    def sample(self, indices: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Build the batch for a batch of events.

        Args:
            indices (torch.Tensor): Event indices, on `self.device`.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]: The batch, on `self.device`.
        """
        raise NotImplementedError()

    @staticmethod
    def make_generator(seed: int, stream: int, device: Union[str, torch.device] = "cpu") -> torch.Generator:
        """Build a generator for one of the independent RNG streams of a seed.

        Args:
            seed (int): The base seed.
            stream (int): The stream (e.g. `BATCH_STREAM`).
            device (Union[str, torch.device]): The generator's device. Defaults to CPU.

        Returns:
            torch.Generator: The seeded generator.
        """
        state = np.random.SeedSequence([seed, stream]).generate_state(1, dtype=np.uint64)[0]
        return torch.Generator(device=device).manual_seed(int(state & 0x7FFF_FFFF_FFFF_FFFF))

    def to(self, device: Union[str, torch.device]) -> AbstractSampler:
        """Move the state to the device the batches will be built on. Meant to be
        called once, before any batch is drawn.

        Args:
            device (Union[str, torch.device]): The target device.

        Returns:
            AbstractSampler: This sampler.
        """
        device = torch.device(device)
        for name in self._state_names:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(device))

        self.device = device
        self._generator = self.make_generator(self._seed, self.BATCH_STREAM, device)
        return self

    def _register_state(self, name: str, value: Union[torch.Tensor, np.ndarray, KeySet, None]):
        """Set an attribute and track it as state, moved by `to`.

        Args:
            name (str): The attribute name.
            value (Union[torch.Tensor, np.ndarray, KeySet, None]): The value; a numpy
                array is converted to a tensor.
        """
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(np.ascontiguousarray(value))
        setattr(self, name, value)
        if name not in self._state_names:
            self._state_names.append(name)

    @property
    def _float_dtype(self) -> torch.dtype:
        # MPS has no float64
        return torch.float32 if self.device.type == "mps" else torch.float64

    def _randint(self, high: int, size: Tuple[int, ...]) -> torch.Tensor:
        """Draw uniform integers.

        Args:
            high (int): Exclusive upper bound.
            size (Tuple[int, ...]): Output shape.

        Returns:
            torch.Tensor: Integers in `[0, high)`, int64.
        """
        return torch.randint(high, size, generator=self._generator, device=self.device)

    def _rand(self, size: Tuple[int, ...]) -> torch.Tensor:
        """Draw uniform floats.

        Args:
            size (Tuple[int, ...]): Output shape.

        Returns:
            torch.Tensor: Floats in `[0, 1)`.
        """
        return torch.rand(size, generator=self._generator, device=self.device, dtype=self._float_dtype)

    def _rand_below(self, highs: torch.Tensor) -> torch.Tensor:
        """Draw one uniform integer per element, each with its own upper bound.

        Args:
            highs (torch.Tensor): Exclusive upper bounds.

        Returns:
            torch.Tensor: Integers in `[0, highs[i])` (0 where `highs[i] <= 0`), int64.
        """
        draws = (self._rand(tuple(highs.shape)) * highs.to(self._float_dtype)).long()
        # Float rounding could push `u * high` up to `high` itself
        return torch.minimum(draws, (highs - 1).clamp(min=0))

    def _draw_negatives(
        self,
        owners: torch.Tensor,
        k: int,
        n_candidates: int,
        seen: Optional[KeySet] = None,
        exclude: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Draw, with replacement, `k` negatives per row, never one of the values
        `seen` holds for the row's owner nor the row's own `exclude` value.
        Rejected draws are redrawn until none is left.

        Args:
            owners (torch.Tensor): The `seen` group of each row, shape `(n_rows,)`.
            k (int): Negatives per row.
            n_candidates (int): Negatives are drawn from `[0, n_candidates)`.
            seen (KeySet, optional): Values excluded per owner.
            exclude (torch.Tensor, optional): One value excluded per row; one
                outside `[0, n_candidates)` excludes nothing.

        Returns:
            torch.Tensor: The negatives, int64, shape `(n_rows, k)`.

        Raises:
            ValueError: If some row has no candidate left.
        """
        n_rows = owners.numel()
        if n_rows == 0 or k == 0:
            return torch.empty((n_rows, k), dtype=torch.long, device=self.device)

        self._check_negative_pool(owners, n_candidates, seen, exclude)

        def rejected(cand, row_owners, row_exclude):
            bad = torch.zeros(cand.shape, dtype=torch.bool, device=cand.device)
            if seen is not None:
                bad |= seen.contains(row_owners, cand)
            if row_exclude is not None:
                bad |= cand == row_exclude
            return bad

        out = self._randint(n_candidates, (n_rows, k))
        rows, cols = torch.nonzero(
            rejected(out, owners[:, None], None if exclude is None else exclude[:, None]), as_tuple=True
        )
        while rows.numel():
            cand = self._randint(n_candidates, (rows.numel(),))
            out[rows, cols] = cand
            still_bad = rejected(cand, owners[rows], None if exclude is None else exclude[rows])
            rows, cols = rows[still_bad], cols[still_bad]

        return out

    @staticmethod
    def _check_negative_pool(
        owners: torch.Tensor,
        n_candidates: int,
        seen: Optional[KeySet],
        exclude: Optional[torch.Tensor]
    ):
        """Check that every row of `_draw_negatives` has a candidate left.

        Args:
            owners (torch.Tensor): The `seen` group of each row.
            n_candidates (int): Number of candidates.
            seen (KeySet, optional): Values excluded per owner.
            exclude (torch.Tensor, optional): One value excluded per row.

        Raises:
            ValueError: If some row has no candidate left.
        """
        # With two candidates left for every owner, every row keeps one: no check (nor device sync) needed
        if n_candidates - (seen.max_count if seen is not None else 0) > 1:
            return

        n_left = torch.full(owners.shape, n_candidates, dtype=torch.long, device=owners.device)
        if seen is not None:
            n_left -= seen.counts[owners]
        if exclude is not None:
            # An excluded value already seen doesn't shrink the pool any further
            shrinks = (exclude >= 0) & (exclude < n_candidates)
            if seen is not None:
                shrinks &= ~seen.contains(owners, exclude)
            n_left -= shrinks.long()

        if bool((n_left <= 0).any()):
            raise ValueError("No candidate left to sample negatives from.")


class InteractionSampler(AbstractSampler):
    """Base of the samplers drawing from a split's interactions.

    Args:
        train_dict (Dict[int, Dict[int, float]]): Private ratings dict (`user -> {item: rating}`).
        transactions (int): Number of events.
        **kwargs (Any): Forwarded to `AbstractSampler`.
    """

    type = SamplerType.INTERACTIONS

    def __init__(self, train_dict: Dict[int, Dict[int, float]], transactions: int, **kwargs: Any):
        super().__init__(**kwargs)

        self.events = transactions
        self._indexed_ratings = train_dict

    def _build_interaction_state(self, with_ratings: bool = False):
        """Register the state most interaction samplers share:

        - `_ui_indptr`, `_ui_indices` (and `_ui_ratings`): every user's sorted rated
          items, as a CSR;
        - `_ui_seen`: the same pairs as a `KeySet`, to exclude them from negatives;
        - `_ui_usable`, `_ui_usable_users`: the users with at least one rated and
          one unrated item.

        Args:
            with_ratings (bool): Also register `_ui_ratings` (float32). Defaults to False.
        """
        rated = {u: sorted(items) for u, items in self._indexed_ratings.items()}

        degree = np.zeros(self._nusers, dtype=np.int64)
        for u, items in rated.items():
            degree[u] = len(items)
        indptr = np.concatenate(([0], np.cumsum(degree)))

        users = np.repeat(np.arange(self._nusers, dtype=np.int64), degree)
        indices = np.fromiter(
            chain.from_iterable(rated.get(u, ()) for u in range(self._nusers)),
            dtype=np.int64, count=int(indptr[-1])
        )

        self._register_state("_ui_indptr", indptr)
        self._register_state("_ui_indices", indices)
        self._register_state("_ui_seen", KeySet.from_pairs(users, indices, self._nusers, self._nitems))

        if with_ratings:
            ratings = np.fromiter(
                (self._indexed_ratings[u][i] for u, i in zip(users.tolist(), indices.tolist())),
                dtype=np.float32, count=len(indices)
            )
            self._register_state("_ui_ratings", ratings)

        usable = (degree > 0) & (degree < self._nitems)
        self._register_state("_ui_usable", usable)
        self._register_state("_ui_usable_users", np.flatnonzero(usable).astype(np.int64))
        self._n_usable_users = int(usable.sum())

    def _register_usable_pairs(self):
        """Register `_pair_users`/`_pair_items`: every (user, rated item) pair of the
        users with at least one unrated item. Needs `_build_interaction_state`."""
        degree = np.diff(self._ui_indptr.numpy())
        users = np.repeat(np.arange(self._nusers, dtype=np.int64), degree)
        keep = (degree < self._nitems)[users]

        self._register_state("_pair_users", users[keep])
        self._register_state("_pair_items", self._ui_indices.numpy()[keep])

    def _draw_users(self) -> torch.Tensor:
        """Draw one user per event, uniformly, once at construction time.

        Returns:
            torch.Tensor: Private user indices, int64, shape `(events,)`.
        """
        return torch.randint(self._nusers, (self.events,), generator=self._init_generator)

    def _redraw_unusable_users(self, users: torch.Tensor) -> torch.Tensor:
        """Replace every unusable user (no rated or no unrated item) with a random usable one.

        Args:
            users (torch.Tensor): Private user indices.

        Returns:
            torch.Tensor: The users, unusable ones replaced.

        Raises:
            ValueError: If no user is usable.
        """
        if self._n_usable_users == self._nusers:
            return users
        if self._n_usable_users == 0:
            raise ValueError("No user has both a rated and an unrated item to sample from.")

        # Drawn for every row, avoiding a device sync, and kept only where needed
        redrawn = self._ui_usable_users[self._randint(self._n_usable_users, tuple(users.shape))]
        return torch.where(self._ui_usable[users], users, redrawn)

    def _draw_rated(self, users: torch.Tensor) -> torch.Tensor:
        """Draw one random rated item per user.

        Args:
            users (torch.Tensor): Private user indices, each with a rated item.

        Returns:
            torch.Tensor: The items' CSR positions (index `_ui_indices`/`_ui_ratings` with them).
        """
        start = self._ui_indptr[users]
        return start + self._rand_below(self._ui_indptr[users + 1] - start)

    def _draw_unrated(self, users: torch.Tensor, k: int) -> torch.Tensor:
        """Draw `k` random unrated items per user, with replacement.

        Args:
            users (torch.Tensor): Private user indices, each with an unrated item.
            k (int): Items per user.

        Returns:
            torch.Tensor: The items, int64, shape `(len(users), k)`.
        """
        return self._draw_negatives(users, k, self._nitems, seen=self._ui_seen)


class SessionSampler(AbstractSampler):
    """Base of the samplers drawing from `Sessions`' flat item tape, sorted by
    user, session and timestamp. A sequence never crosses its boundary segment:
    the user's history (FLAT) or the session (SESSION_ONLY).

    Args:
        flat_items (np.ndarray): Item per tape position.
        flat_users (np.ndarray): Owning (private) user per tape position.
        flat_session (np.ndarray): Owning (private) session per tape position.
        user_offsets (np.ndarray): Per-user boundaries over the tape.
        session_offsets (np.ndarray): Per-session boundaries over the tape.
        sparse (csr_matrix): Train interactions, the items excluded from negatives.
        strategy (SessionStrategy): FLAT or SESSION_ONLY. Defaults to FLAT.
        max_seq_len (int): Maximum sequence length. Defaults to 50.
        neg_samples (int): Negatives per target, 0 for none. Defaults to 0.
        exclude_seen (bool): Whether negatives exclude every item the user has in
            `sparse`, not only the row's own target. Defaults to True.
        **kwargs (Any): Forwarded to `AbstractSampler`.
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
        exclude_seen: bool = True,
        **kwargs: Any
    ):
        super().__init__(**kwargs)

        self._flat_items = np.asarray(flat_items)
        self._flat_users = np.asarray(flat_users)
        self._flat_session = np.asarray(flat_session)
        self._sparse = sparse

        self.strategy = SessionStrategy(strategy)
        self.max_seq_len = max_seq_len

        self._neg_samples = neg_samples
        self._exclude_seen = exclude_seen
        self._padding_token = self._nitems

        session_only = self.strategy == SessionStrategy.SESSION_ONLY
        self._boundaries = np.asarray(session_offsets if session_only else user_offsets, dtype=np.int64)
        self._segment_of = self._flat_session if session_only else self._flat_users

        self._valid_target_indices = self._compute_valid_targets()
        self.events = len(self._valid_target_indices)

        self._build_session_state()

    def _build_session_state(self):
        """Register the tape's items and users (`_tape_items`, `_tape_users`, int32
        when they fit) and, if negatives exclude seen items, each user's items in
        `sparse` as a `KeySet` (`_tape_seen`)."""
        def compact(array):
            fits = array.size == 0 or int(array.max()) < np.iinfo(np.int32).max
            return array.astype(np.int32 if fits else np.int64, copy=False)

        self._register_state("_tape_items", compact(self._flat_items))
        self._register_state("_tape_users", compact(self._flat_users))

        seen = None
        if self._neg_samples > 0 and self._exclude_seen:
            coo = self._sparse.tocoo()
            seen = KeySet.from_pairs(coo.row, coo.col, self._sparse.shape[0], self._nitems)
        self._register_state("_tape_seen", seen)

    @staticmethod
    def _ragged_arange(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenate `arange(counts[g])` over every group `g`.

        Args:
            counts (np.ndarray): Number of rows per group.

        Returns:
            Tuple[np.ndarray, np.ndarray]: `(group, offset)` of every row, int64.
        """
        counts = np.asarray(counts, dtype=np.int64)
        group = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        group_first_row = np.cumsum(counts) - counts
        return group, np.arange(len(group), dtype=np.int64) - group_first_row[group]

    def _segment_spans(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the tape span of every boundary segment.

        Returns:
            Tuple[np.ndarray, np.ndarray]: `(starts, ends)`, ends exclusive.
        """
        return self._boundaries[:-1], self._boundaries[1:]

    def _compute_valid_targets(self, horizon: int = 1) -> np.ndarray:
        """Find the tape positions `p` with at least one predecessor in their
        segment and room for `horizon` targets from `p` on.

        Args:
            horizon (int): Consecutive targets per position. Defaults to 1.

        Returns:
            np.ndarray: The valid target positions, ascending.
        """
        starts, ends = self._segment_spans()
        first = starts + 1
        segment, offset = self._ragged_arange(np.maximum(ends - horizon + 1 - first, 0))
        return first[segment] + offset

    def _register_target_events(self):
        """Register `_ev_target`/`_ev_start`, per event: the target position and
        where its input sequence starts (at most `max_seq_len` items back, never
        before its segment's start)."""
        targets = self._valid_target_indices.astype(np.int64, copy=False)
        segment_starts = self._boundaries[self._segment_of[targets]]

        self._register_state("_ev_target", targets)
        self._register_state("_ev_start", np.maximum(segment_starts, targets - self.max_seq_len))

    def _padded_batch(self, starts: torch.Tensor, ends: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read one sequence per row from the tape, left-aligned and right-padded.

        Args:
            starts (torch.Tensor): First tape position per row.
            ends (torch.Tensor): Exclusive last tape position per row, at most
                `max_seq_len` after its start.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The sequences, int64, shape
                `(batch, max_seq_len)`, and their real lengths.
        """
        offsets = torch.arange(self.max_seq_len, device=starts.device)
        lengths = ends - starts
        real = offsets < lengths[:, None]

        positions = torch.where(real, starts[:, None] + offsets, 0)
        seq = torch.where(real, self._tape_items[positions].long(), self._padding_token)
        return seq, lengths

    def _draw_session_negatives(self, owners: torch.Tensor, k: int, exclude: torch.Tensor) -> torch.Tensor:
        """Draw `k` negative items per row, never the row's `exclude` item nor, if
        `exclude_seen`, any item its owner has seen.

        Args:
            owners (torch.Tensor): Owning user per row.
            k (int): Negatives per row.
            exclude (torch.Tensor): Item excluded per row (the padding token excludes nothing).

        Returns:
            torch.Tensor: The negatives, int64, shape `(rows, k)`.
        """
        return self._draw_negatives(owners.long(), k, self._nitems, seen=self._tape_seen, exclude=exclude)


class SideInfoSampler(AbstractSampler):
    """Base of the samplers drawing from side information (e.g. a knowledge graph)
    instead of interactions. Wired in through a recommender's own
    `side_info_sampler_config`, and zipped with the interaction batches by
    `CombinedDataLoader`."""

    type = SamplerType.SIDE_INFO


class KeySet:
    """A set of `(group, value)` pairs (e.g. the items each user has seen),
    queried in batch on any device. Each pair is stored as the key
    `group * span + value`, kept sorted, so a batch of membership queries is a
    single `torch.searchsorted`.

    Args:
        keys (torch.Tensor): Sorted, unique int64 keys.
        counts (torch.Tensor): Number of values per group.
        span (int): Exclusive upper bound of the values.
    """

    def __init__(self, keys: torch.Tensor, counts: torch.Tensor, span: int):
        self.keys = keys
        self.counts = counts
        self.span = span
        self.max_count = int(counts.max()) if counts.numel() else 0

    @classmethod
    def from_pairs(cls, groups: np.ndarray, values: np.ndarray, n_groups: int, span: int) -> KeySet:
        """Build the set, on CPU, dropping duplicate pairs.

        Args:
            groups (np.ndarray): Group per pair, in `[0, n_groups)`.
            values (np.ndarray): Value per pair, in `[0, span)`.
            n_groups (int): Number of groups.
            span (int): Exclusive upper bound of the values.

        Returns:
            KeySet: The set.
        """
        keys = np.unique(
            np.asarray(groups, dtype=np.int64) * span 
            + np.asarray(values, dtype=np.int64)
        )
        counts = (
            np.bincount(keys // span, minlength=n_groups) 
            if span else np.zeros(n_groups, dtype=np.int64)
        )
        return cls(torch.from_numpy(keys), torch.from_numpy(counts.astype(np.int64)), span)

    def contains(self, groups: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Check, element-wise, which `(group, value)` pairs are in the set.

        Args:
            groups (torch.Tensor): Groups, broadcastable with `values`.
            values (torch.Tensor): Values; one outside `[0, span)` is never in the set.

        Returns:
            torch.Tensor: Bool mask of the pairs in the set.
        """
        values = values.long()
        in_range = (values >= 0) & (values < self.span)
        if not self.keys.numel():
            return torch.zeros_like(in_range)

        query = groups.long() * self.span + values
        pos = torch.searchsorted(self.keys, query).clamp_(max=self.keys.numel() - 1)
        return (self.keys[pos] == query) & in_range

    def to(self, device: torch.device) -> KeySet:
        """Copy the set to a device.

        Args:
            device (torch.device): The target device.

        Returns:
            KeySet: The set, on `device`.
        """
        moved = KeySet.__new__(KeySet)
        moved.keys, moved.counts = self.keys.to(device), self.counts.to(device)
        moved.span, moved.max_count = self.span, self.max_count
        return moved
