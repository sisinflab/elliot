from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import time
from bisect import bisect_left
from itertools import chain
import numpy as np
import torch
from abc import ABC, abstractmethod
from logging import LoggerAdapter
from scipy.sparse import csr_matrix
from tqdm import tqdm

from elliot.utils import logging as elog
from elliot.utils.enums import SamplerMaterialization, SamplerType, SessionStrategy


class AbstractSampler(ABC):
    """Base class every sampler registered in `sampler_registry` implements.

    A concrete sampler is always built from two orthogonal mixins, combined via
    multiple inheritance: a domain mixin, which sets `type` and owns whatever
    domain-specific state/helpers it needs (e.g. `_InteractionDataMixin`,
    `_SessionDataMixin`, `_SideInfoMixin`), and a materialization-strategy mixin,
    which sets `materialization` and owns how `sample(it)`'s stream becomes a
    `torch.utils.data.Dataset` (`_TraditionalMixin` or `_PipelineMixin`). Neither
    axis knows about the other, so any domain can be combined with either strategy.

    Args:
        users (List[int], optional): Private user indices in this split's domain.
            Defaults to `[]` - a side-info sampler (see `PipelineSideInfoSampler`,
            `TraditionalSideInfoSampler`) draws from its own source and never
            receives one.
        items (List[int], optional): Private item indices in this split's domain.
            Defaults to `[]`, for the same reason as `users`.
        n_users (int): Total number of users. Defaults to 0.
        n_items (int): Total number of items. Defaults to 0.
        seed (int): Random seed for reproducibility.
        logger (LoggerAdapter, optional): Logging instance. Defaults to None, building
            a fresh one via `elliot.utils.logging.get_logger`.
        **kwargs (Any): Unused; absorbs any extra keyword arguments forwarded by
            `sampler_registry.get()` that a concrete sampler doesn't itself declare.
    """

    type: SamplerType
    materialization: SamplerMaterialization

    # Uniform floats drawn per refill of the single-integer `_r_int` buffer
    RAND_BUFFER_SIZE: int = 8192

    # Most negatives (rows * k) `_draw_negatives` draws one at a time instead of vectorized
    SMALL_NEGATIVES_SIZE: int = 64

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

        # Initializing variables
        self._users = users if users is not None else []
        self._nusers = n_users
        self._items = items if items is not None else []
        self._nitems = n_items

        # This sampler's own RNG: every draw comes from it, never from a global state
        self.reseed(seed)

        self.events: int = 0

    def reseed(self, seed: int):
        """(Re)initialize this sampler's own torch RNG, discarding any buffered draw.

        Called once at construction time and again inside every dataloader worker
        (see `elliot.dataset.samplers.base_data._seed_worker`), since each worker
        receives an identical copy of the parent's sampler and RNG state.

        Args:
            seed (int): The new seed.
        """
        self._generator = torch.Generator().manual_seed(seed)
        self._rand_buffer: List[float] = []

    def _r_int(
        self,
        low: int,
        high: Optional[int] = None,
        size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[int, torch.Tensor]:
        """Uniform random integers in `[low, high)`, with `numpy.random.randint`'s
        call convention: a single argument `n` is taken as the exclusive upper bound,
        so `_r_int(n)` draws from `[0, n)`.

        A single draw pops one uniform float off a buffer refilled `RAND_BUFFER_SIZE`
        draws at a time, so the per-call overhead stays far below a torch call's.
        That single draw is a plain `int`, cheap to index Python containers with and
        usable as a dict key (a 0-d tensor is neither).

        Args:
            low (int): Lowest integer drawn, or the exclusive upper bound when `high`
                is None.
            high (int, optional): Exclusive upper bound. Defaults to None.
            size (int or Tuple[int, ...], optional): Output shape. Defaults to None,
                drawing a single integer.

        Returns:
            Union[int, torch.Tensor]: A single int when `size` is None, else an int64
                tensor shaped `size`.
        """
        if high is None:
            low, high = 0, low

        if size is not None:
            size = (size,) if isinstance(size, int) else tuple(size)
            return torch.randint(low, high, size, generator=self._generator)

        if not self._rand_buffer:
            self._rand_buffer = torch.rand(
                self.RAND_BUFFER_SIZE, dtype=torch.float64, generator=self._generator
            ).tolist()

        span = high - low
        # Guard against float rounding pushing `u * span` up to `span` itself
        return low + min(int(self._rand_buffer.pop() * span), span - 1)

    def _r_choice(self, a: int, size: int, replace: bool = True) -> torch.Tensor:
        """Uniform random draw of `size` integers from `[0, a)`, with
        `numpy.random.choice`'s call convention for an integer `a`.

        Args:
            a (int): Exclusive upper bound of the drawn integers.
            size (int): Number of integers to draw.
            replace (bool): Whether to draw with replacement. Defaults to True.

        Returns:
            torch.Tensor: The drawn int64 integers, shape `(size,)`.

        Note:
            Without replacement the draw permutes the whole `[0, a)`, costing
            `O(a)` whatever `size`: keep `a` small on a per-event path.
        """
        if replace:
            return torch.randint(a, (size,), generator=self._generator)
        return torch.randperm(a, generator=self._generator)[:size]

    def _r_sample(self, n: int, k: int) -> torch.Tensor:
        """Draw `k` distinct indices from `[0, n)`, uniformly at random.

        Args:
            n (int): Number of indices to draw from.
            k (int): Number of indices to draw.

        Returns:
            torch.Tensor: The drawn int64 indices, in random order, shape `(k,)`.

        Note:
            The draw permutes the whole `[0, n)`, costing `O(n)` whatever `k`:
            keep `n` small on a per-event path.
        """
        return torch.randperm(n, generator=self._generator)[:k]

    def _seen_items(self, user: int) -> Sequence[int]:
        """`user`'s seen items, sorted: the ones `_sample_negative*` never draws.
        Defined by the domain mixins that sample negatives for a user.

        Args:
            user (int): Private user index.

        Returns:
            Sequence[int]: The sorted item ids excluded from `user`'s negatives.
        """
        raise NotImplementedError()

    def _sample_negative(self, user: int) -> int:
        """Uniformly sample one negative item for `user`, excluding its seen items.

        Args:
            user (int): Private user index.

        Returns:
            int: The sampled negative item id.
        """
        return self._draw_negative(self._seen_items(user), self._nitems)

    def _sample_negatives(self, user: int, k: int, exclude_item: Optional[int] = None) -> torch.Tensor:
        """Uniformly sample, with replacement, `k` negative items for `user`,
        excluding its seen items and, optionally, `exclude_item`.

        Args:
            user (int): Private user index.
            k (int): Number of negatives to sample.
            exclude_item (int, optional): An extra item id to exclude (e.g. the
                positive target). Defaults to None.

        Returns:
            torch.Tensor: The sampled negative item ids, shape `(k,)`.
        """
        exclude_items = None if exclude_item is None else [exclude_item]
        return self._sample_negatives_rows(user, k, n_rows=1, exclude_items=exclude_items)[0]

    def _sample_negatives_rows(
        self,
        user: int,
        k: int,
        n_rows: int,
        exclude_items: Optional[Union[np.ndarray, torch.Tensor, Sequence[int]]] = None
    ) -> torch.Tensor:
        """Uniformly sample, with replacement, `k` negative items for each of `n_rows`
        rows of `user`, excluding its seen items and, optionally, each row's own
        `exclude_items[row]`.

        Args:
            user (int): Private user index.
            k (int): Number of negatives to sample per row.
            n_rows (int): Number of rows to sample.
            exclude_items (np.ndarray, torch.Tensor or Sequence[int], optional): One
                extra item id to exclude per row, shape `(n_rows,)`. Defaults to None.

        Returns:
            torch.Tensor: The sampled negative item ids, shape `(n_rows, k)`.
        """
        return self._draw_negatives(self._seen_items(user), self._nitems, k, n_rows, exclude_items)

    def _draw_negative(self, seen: Sequence[int], n_candidates: int, exclude: Optional[int] = None) -> int:
        """Rejection sampling of one negative from `[0, n_candidates)`, excluding
        `seen` and, optionally, `exclude`.

        Args:
            seen (Sequence[int]): Sorted ids to exclude.
            n_candidates (int): Candidates are drawn from `[0, n_candidates)`.
            exclude (int, optional): An extra id to exclude. Defaults to None.

        Returns:
            int: The sampled negative id.

        Raises:
            ValueError: If no candidate is left to draw.
        """
        self._check_negative_pool(seen, n_candidates, None if exclude is None else [exclude])

        n_seen = len(seen)
        while True:
            cand = self._r_int(n_candidates)
            if cand == exclude:
                continue
            # Binary search since seen is sorted
            pos = bisect_left(seen, cand)
            if pos < n_seen and seen[pos] == cand:
                continue
            return cand

    def _draw_negatives(
        self,
        seen: Sequence[int],
        n_candidates: int,
        k: int,
        n_rows: int = 1,
        exclude_items: Optional[Union[np.ndarray, torch.Tensor, Sequence[int]]] = None
    ) -> torch.Tensor:
        """Rejection sampling, with replacement, of `k` negatives from
        `[0, n_candidates)` for each of `n_rows` rows, excluding `seen` and,
        optionally, each row's own `exclude_items[row]`.

        Run by `_draw_negatives_loop` for small requests (up to
        `SMALL_NEGATIVES_SIZE` slots) and by `_draw_negatives_vectorized` otherwise.

        Args:
            seen (Sequence[int]): Sorted ids excluded from every row.
            n_candidates (int): Candidates are drawn from `[0, n_candidates)`.
            k (int): Number of negatives to sample per row.
            n_rows (int): Number of rows to sample. Defaults to 1.
            exclude_items (np.ndarray, torch.Tensor or Sequence[int], optional): One
                extra id to exclude per row, shape `(n_rows,)`. Defaults to None.

        Returns:
            torch.Tensor: The sampled negative ids, shape `(n_rows, k)`.

        Raises:
            ValueError: If some row has no candidate left to draw.
        """
        if n_rows == 0 or k == 0:
            return torch.empty((n_rows, k), dtype=torch.long)

        self._check_negative_pool(seen, n_candidates, exclude_items)

        if n_rows * k <= self.SMALL_NEGATIVES_SIZE:
            return self._draw_negatives_loop(seen, n_candidates, k, n_rows, exclude_items)
        return self._draw_negatives_vectorized(seen, n_candidates, k, n_rows, exclude_items)

    def _draw_negatives_loop(
        self,
        seen: Sequence[int],
        n_candidates: int,
        k: int,
        n_rows: int,
        exclude_items: Optional[Union[np.ndarray, torch.Tensor, Sequence[int]]]
    ) -> torch.Tensor:
        """`_draw_negatives` as a plain loop of `_draw_negative` calls: for a handful
        of slots, far cheaper than the fixed overhead of torch calls.

        Args:
            seen (Sequence[int]): Sorted ids excluded from every row.
            n_candidates (int): Candidates are drawn from `[0, n_candidates)`.
            k (int): Number of negatives to sample per row.
            n_rows (int): Number of rows to sample.
            exclude_items (np.ndarray, torch.Tensor or Sequence[int], optional): One
                extra id to exclude per row, shape `(n_rows,)`.

        Returns:
            torch.Tensor: The sampled negative ids, shape `(n_rows, k)`.
        """
        if exclude_items is None:
            excluded = [None] * n_rows
        else:
            excluded = exclude_items.tolist() if hasattr(exclude_items, "tolist") else list(exclude_items)

        negatives = [
            self._draw_negative(seen, n_candidates, row_excluded)
            for row_excluded in excluded for _ in range(k)
        ]
        return torch.tensor(negatives, dtype=torch.long).view(n_rows, k)

    def _draw_negatives_vectorized(
        self,
        seen: Sequence[int],
        n_candidates: int,
        k: int,
        n_rows: int,
        exclude_items: Optional[Union[np.ndarray, torch.Tensor, Sequence[int]]]
    ) -> torch.Tensor:
        """`_draw_negatives` vectorized: every slot is drawn at once, then only the
        rejected ones are redrawn, round after round.

        Args:
            seen (Sequence[int]): Sorted ids excluded from every row.
            n_candidates (int): Candidates are drawn from `[0, n_candidates)`.
            k (int): Number of negatives to sample per row.
            n_rows (int): Number of rows to sample.
            exclude_items (np.ndarray, torch.Tensor or Sequence[int], optional): One
                extra id to exclude per row, shape `(n_rows,)`.

        Returns:
            torch.Tensor: The sampled negative ids, shape `(n_rows, k)`.
        """
        seen_items = torch.as_tensor(seen, dtype=torch.long)
        n_seen = len(seen_items)
        if exclude_items is not None:
            exclude_items = torch.as_tensor(exclude_items, dtype=torch.long)

        def rejected(cand: torch.Tensor, excluded: Optional[torch.Tensor]) -> torch.Tensor:
            """Mask of the candidates that are seen ids or their row's excluded id."""
            bad = torch.zeros(cand.shape, dtype=torch.bool)
            if n_seen:
                # Binary search since seen_items is sorted
                pos = torch.searchsorted(seen_items, cand).clamp_(max=n_seen - 1)
                bad |= seen_items[pos] == cand
            if excluded is not None:
                bad |= cand == excluded
            return bad

        out = torch.randint(n_candidates, (n_rows, k), generator=self._generator)
        rows, cols = torch.nonzero(
            rejected(out, None if exclude_items is None else exclude_items[:, None]), as_tuple=True
        )

        # Redraw only the rejected slots, until every slot holds a valid negative
        while len(rows):
            cand = torch.randint(n_candidates, (len(rows),), generator=self._generator)
            out[rows, cols] = cand
            still_bad = rejected(cand, None if exclude_items is None else exclude_items[rows])
            rows, cols = rows[still_bad], cols[still_bad]

        return out

    @staticmethod
    def _check_negative_pool(
        seen: Sequence[int],
        n_candidates: int,
        exclude_items: Optional[Union[np.ndarray, torch.Tensor, Sequence[int]]] = None
    ):
        """Make sure every row keeps at least one valid negative to draw.

        Args:
            seen (Sequence[int]): Sorted ids excluded from every row.
            n_candidates (int): Candidates are drawn from `[0, n_candidates)`.
            exclude_items (np.ndarray, torch.Tensor or Sequence[int], optional): One
                extra id excluded per row. Defaults to None.

        Raises:
            ValueError: If some row has no candidate left to draw.
        """
        # With at least two candidates left every row keeps one, whatever it excludes
        n_left = n_candidates - len(seen)
        if n_left > 1:
            return

        no_pool = n_left <= 0
        if exclude_items is not None and not no_pool:
            # An excluded id already among the seen ones doesn't shrink the pool any further
            excluded = np.asarray(exclude_items).reshape(-1)
            no_pool = bool(np.any(~np.isin(excluded, seen)))
        if no_pool:
            raise ValueError("No candidate left to sample negatives from.")

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
        """Optional hook: `sample`'s evaluation-time counterpart. Default is a no-op.

        Args:
            it (int): Event index.

        Returns:
            Any: The sample, whose shape is defined by the concrete sampler.
        """
        pass


class _TraditionalMixin:
    """Materialization strategy: eagerly builds the whole event stream at once,
    via `sample_full()`, for consumption as a single in-memory `TensorDataset`.
    """

    materialization = SamplerMaterialization.TRADITIONAL

    def sample_full(self, val: bool = False) -> List[Any]:
        """Build every sample in the event stream at once, in event order.

        `build_dataset` calls it once and the resulting dataset is cached, so every
        epoch replays these exact samples (negatives included): the training
        dataloader only reshuffles their order. A sampler whose draws must change
        across epochs belongs to the pipeline materialization instead.

        Args:
            val (bool): If True, use `sample_eval` instead of `sample`.
                Defaults to False.

        Returns:
            List[Any]: The full sample stream.
        """
        start = time.time()

        iter_data = tqdm(
            range(self.events),
            total=self.events,
            desc="Sampling",
            leave=False
        )
        samples = []

        # Pick sample() or its eval-time counterpart
        sample_fn = self.sample if not val else self.sample_eval

        for it in iter_data:
            output = sample_fn(it)

            # A hook may explode one sample into several (e.g. windowed sequences)
            if isinstance(output, list):
                samples.extend(output)
            else:
                samples.append(output)

        end = time.time()

        self.logger.debug(
            "Completed sampling",
            extra={"context": {"duration_sec": round(end - start, 4), "events": len(samples)}}
        )

        return samples


class _PipelineMixin:
    """Materialization strategy: lazily sampled, `sample(it)` called on demand,
    once per dataset index, by `PipelineDataset`.
    """

    materialization = SamplerMaterialization.PIPELINE

    def collate_fn(self, batch: List[Any]) -> Tuple[torch.Tensor, ...]:
        """Stack each tuple position of a batch of samples into its own tensor.

        The batch needs no shuffling here: the training dataloader already draws
        its samples in a random order.

        Args:
            batch (List[Any]): The batch of samples, all tuples of the same length;
                each tuple position holds either same-shape tensors, plain scalars or
                lists of scalars, consistently across the batch.

        Returns:
            Tuple[torch.Tensor, ...]: One tensor per tuple position: tensors stacked
                along a new batch dimension, scalars (lists concatenated) converted
                into an int64 tensor, or a floating one if any value is a float.
        """
        # Transpose the batch of tuples into one tensor per tuple position
        tensors = []
        for column in zip(*batch):
            if isinstance(column[0], torch.Tensor):
                tensors.append(torch.stack(column))
                continue

            # A sample's list (e.g. a positive plus its negatives) adds one row per value
            if isinstance(column[0], list):
                column = list(chain.from_iterable(column))

            # Integer columns become int64, a float one (e.g. ratings) is not truncated
            tensor = torch.tensor(column)
            tensors.append(tensor if tensor.is_floating_point() else tensor.long())

        return tuple(tensors)


class _InteractionDataMixin:
    """Domain: `SamplerType.INTERACTIONS`. A split's interaction dict, indexed
    for O(1) sampling.

    Args:
        train_dict (Dict[int, Dict[int, float]]): Private-id-keyed ratings dict for
            this split (`user -> {item: rating}`).
        transactions (int): Number of events to sample.
        **kwargs (Any): Forwarded to `AbstractSampler.__init__`.
    """

    type = SamplerType.INTERACTIONS

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

        # Per-user sorted item list and its length, cached for O(1) sampling
        self._ui_dict = {u: sorted(set(self._indexed_ratings[u])) for u in self._indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

        # Per-event pre-drawn users, set by the samplers that draw them up front
        self._sampled_users: List[int] = []

        # Users with at least one rated and one unrated item, built on first need
        # by `_draw_usable_user`
        self._usable_users: Optional[List[int]] = None

    def _draw_usable_user(self) -> Tuple[int, List[int], int]:
        """Uniformly draw a user with at least one rated and one unrated item, as a
        fallback for a pre-drawn user who can't be used (e.g. one who rated every
        item, leaving no valid negative).

        Returns:
            Tuple[int, List[int], int]: The drawn user, its rated items and their count.

        Raises:
            ValueError: If no user has both a rated and an unrated item.
        """
        if self._usable_users is None:
            self._usable_users = [
                u for u, lui in self._lui_dict.items() if 0 < lui < self._nitems
            ]
            if not self._usable_users:
                raise ValueError("No user has both a rated and an unrated item to sample from.")

        u = self._usable_users[self._r_int(len(self._usable_users))]
        return u, self._ui_dict[u], self._lui_dict[u]

    def _usable_pairs(self) -> List[Tuple[int, int]]:
        """Every (user, rated item) pair of the users with at least one unrated item,
        the only ones a negative can be sampled for.

        Returns:
            List[Tuple[int, int]]: The (user, item) pairs.
        """
        return [
            (u, i) 
            for u, items in self._ui_dict.items() if len(items) < self._nitems 
            for i in items
        ]

    def _seen_items(self, user: int) -> List[int]:
        """`user`'s rated items, sorted.

        Args:
            user (int): Private user index.

        Returns:
            List[int]: The sorted item ids excluded from `user`'s negatives.
        """
        return self._ui_dict[user]


class _SessionDataMixin:
    """Domain: `SamplerType.SEQUENTIAL`. `Sessions`' flat item tape, plus the
    boundary bookkeeping and helpers every session/sequential sampler builds on.

    Subclasses receive the same globally sorted (by user, session, timestamp)
    tape regardless of strategy: the only thing that changes between FLAT and
    SESSION_ONLY is which boundary array bounds a sequence (per-user vs.
    per-session), resolved once here.

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
        exclude_seen (bool): Whether sampled negatives exclude every item the user
            has in `sparse`. If False, only each row's own excluded item (e.g. its
            positive target) is, so that every other item is drawn with the same
            probability. Defaults to True.
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
        exclude_seen: bool = True,
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
        self._exclude_seen = exclude_seen
        self._padding_token = self._nitems

        # Which boundary array bounds a sequence: per-session (SESSION_ONLY) or per-user (FLAT)
        self._boundaries = (
            self._session_offsets if self.strategy == SessionStrategy.SESSION_ONLY else self._user_offsets
        )
        self._valid_target_indices = self._compute_valid_targets()

        self.events = len(self._valid_target_indices)

    @staticmethod
    def _ragged_arange(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enumerate `counts[g]` rows for every group `g`, vectorized: the
        concatenation of `arange(counts[g])` over all groups, plus each row's group.

        Args:
            counts (np.ndarray): Non-negative number of rows per group.

        Returns:
            Tuple[np.ndarray, np.ndarray]: `(group, offset)`, parallel int64 arrays
                of length `counts.sum()`: each row's group id and its offset within
                that group, groups in order and offsets ascending within each.
        """
        counts = np.asarray(counts, dtype=np.int64)
        group = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        group_first_row = np.cumsum(counts) - counts
        return group, np.arange(len(group), dtype=np.int64) - group_first_row[group]

    def _segment_spans(self) -> Tuple[np.ndarray, np.ndarray]:
        """Every boundary segment's `[start, end)` flat tape span.

        Returns:
            Tuple[np.ndarray, np.ndarray]: `(starts, ends)`, int64, one per segment.
        """
        boundaries = self._boundaries.astype(np.int64, copy=False)
        return boundaries[:-1], boundaries[1:]

    def _compute_valid_targets(self, horizon: int = 1) -> np.ndarray:
        """A flat position `p` is a valid target iff it has at least one predecessor
        in its boundary segment and the `horizon` items starting at `p` all fit in
        that segment, i.e. `start + 1 <= p <= end - horizon`.

        Args:
            horizon (int): Number of consecutive target items starting at `p`.
                Defaults to 1, plain next-item prediction.

        Returns:
            np.ndarray: The valid target flat tape positions, ascending.
        """
        starts, ends = self._segment_spans()
        first = starts + 1
        counts = np.maximum(ends - horizon + 1 - first, 0)

        segment, offset = self._ragged_arange(counts)
        return first[segment] + offset

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
        return self._padded_slice(max(boundary_start, end_idx - self.max_seq_len), end_idx)

    def _padded_slice(self, start: int, end: int) -> Tuple[torch.Tensor, int]:
        """The flat tape items in `[start, end)`, left-aligned in a `max_seq_len`
        tensor and right-padded with the padding token.

        Args:
            start (int): First flat tape position, inclusive.
            end (int): Last flat tape position, exclusive; `end - start` must not
                exceed `max_seq_len`.

        Returns:
            Tuple[torch.Tensor, int]: The padded sequence tensor (length
                `max_seq_len`) and its true (pre-padding) length.
        """
        # Built in numpy and converted once.
        # The array is freshly allocated, so the tensor never aliases the flat tape
        seq_len = end - start
        seq_array = np.full(self.max_seq_len, self._padding_token, dtype=np.int64)
        seq_array[:seq_len] = self._flat_items[start:end]
        return torch.from_numpy(seq_array), seq_len

    def _seen_items(self, user: int) -> np.ndarray:
        """`user`'s items in `self._sparse`, sorted (none if not `self._exclude_seen`).

        Args:
            user (int): Private user index.

        Returns:
            np.ndarray: The sorted item ids excluded from `user`'s negatives.
        """
        if not self._exclude_seen:
            return self._sparse.indices[:0]
        return self._sparse.indices[self._sparse.indptr[user]:self._sparse.indptr[user + 1]]


class _SideInfoMixin:
    """Domain: `SamplerType.SIDE_INFO`. Draws from a model's side information
    (e.g. a knowledge graph, item/user attributes, ...) rather than its
    interaction data.

    A side-info sampler is wired in separately from the interaction sampler,
    through a recommender's own `side_info_sampler_config`: the two are sampled,
    batched and cached entirely independently, and only zipped together
    (via `CombinedDataLoader`) into the single batch a model's `train_step` sees.
    """

    type = SamplerType.SIDE_INFO


class TraditionalSampler(_TraditionalMixin, _InteractionDataMixin, AbstractSampler):
    """Interaction sampler, eagerly materialized.

    Args:
        train_dict (Dict[int, Dict[int, float]]): Private-id-keyed ratings dict for
            this split (`user -> {item: rating}`).
        transactions (int): Number of events to sample.
        **kwargs (Any): Forwarded to `AbstractSampler.__init__`.
    """


class PipelineSampler(_PipelineMixin, _InteractionDataMixin, AbstractSampler):
    """Interaction sampler, lazily sampled on demand.

    Args:
        train_dict (Dict[int, Dict[int, float]]): Private-id-keyed ratings dict for
            this split (`user -> {item: rating}`).
        transactions (int): Number of events to sample.
        **kwargs (Any): Forwarded to `AbstractSampler.__init__`.
    """


class TraditionalSessionSampler(_TraditionalMixin, _SessionDataMixin, AbstractSampler):
    """Session sampler, eagerly materialized.

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
        exclude_seen (bool): Whether sampled negatives exclude every item the user
            has in `sparse`. If False, only each row's own excluded item (e.g. its
            positive target) is, so that every other item is drawn with the same
            probability. Defaults to True.
        **kwargs (Any): Forwarded to `AbstractSampler.__init__`.
    """


class PipelineSessionSampler(_PipelineMixin, _SessionDataMixin, AbstractSampler):
    """Session sampler, lazily sampled on demand.

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
        exclude_seen (bool): Whether sampled negatives exclude every item the user
            has in `sparse`. If False, only each row's own excluded item (e.g. its
            positive target) is, so that every other item is drawn with the same
            probability. Defaults to True.
        **kwargs (Any): Forwarded to `AbstractSampler.__init__`.
    """


class TraditionalSideInfoSampler(_TraditionalMixin, _SideInfoMixin, AbstractSampler):
    """Side-info sampler, eagerly materialized.

    Args:
        seed (int): Random seed for reproducibility.
        logger (LoggerAdapter, optional): Logging instance. Defaults to None.
        **kwargs (Any): Forwarded to `AbstractSampler.__init__`.
    """


class PipelineSideInfoSampler(_PipelineMixin, _SideInfoMixin, AbstractSampler):
    """Side-info sampler, lazily sampled on demand.

    Args:
        seed (int): Random seed for reproducibility.
        logger (LoggerAdapter, optional): Logging instance. Defaults to None.
        **kwargs (Any): Forwarded to `AbstractSampler.__init__`.
    """
