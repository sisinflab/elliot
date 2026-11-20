from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch

from elliot.dataset.samplers.base_sampler import PipelineSessionSampler
from elliot.utils.registry import sampler_registry


@sampler_registry.register()
class SequentialSampler(PipelineSessionSampler):
    """Next-item prediction: (sequence, length, target[, negatives]).

    One event per valid target position: the target's own prefix (its last
    `max_seq_len` predecessors within the boundary segment) is the input, and the
    model is supervised on that single position.

    Args:
        target_len (int): Number of consecutive future items predicted from a
            single context window. Defaults to 1, which keeps `target` a scalar
            and `negatives` shaped `(neg_samples,)`, exactly as before;
            `target_len > 1` shapes them `(target_len,)` and
            `(target_len, neg_samples)` instead.
        **params (Any): Forwarded to `PipelineSessionSampler.__init__`.
    """

    def __init__(self, target_len: int = 1, **params: Any):
        # Set before `super().__init__`, which computes the valid targets from it
        self.target_len = target_len
        super().__init__(**params)

    def _compute_valid_targets(self) -> np.ndarray:
        """A flat position is a valid target window start iff it has at least one
        predecessor in its boundary segment and the `target_len` items starting
        there don't run past the segment's end.

        Returns:
            np.ndarray: The valid target flat tape positions.
        """
        return super()._compute_valid_targets(horizon=self.target_len)

    def sample(self, it: int) -> Tuple[torch.Tensor, ...]:
        """Build the (sequence, length, target[, negatives]) sample for event `it`.

        Args:
            it (int): Event index.

        Returns:
            Tuple[torch.Tensor, ...]: `(seq_tensor, seq_len, target)`, plus a
                trailing `negatives` tensor when `self.neg_samples > 0`. `target`
                is a scalar (and `negatives` shaped `(neg_samples,)`) when
                `self.target_len == 1`; otherwise `target` is shaped
                `(target_len,)` (and `negatives` `(target_len, neg_samples)`),
                covering the `target_len` consecutive items starting at this
                flat position.
        """
        target_idx = int(self._valid_target_indices[it])
        owner_user = int(self._flat_users[target_idx])

        # The target(s) are the item(s) at this flat position; context is everything before it
        seq_tensor, seq_len = self._build_padded_sequence(
            target_idx, self._boundary_start_of(target_idx)
        )

        if self.target_len == 1:
            target_item = int(self._flat_items[target_idx])
            target_tensor = torch.tensor(target_item, dtype=torch.long)
        else:
            target_items = self._flat_items[target_idx:target_idx + self.target_len]
            target_tensor = torch.tensor(target_items, dtype=torch.long)

        ret = [seq_tensor, torch.tensor(seq_len, dtype=torch.long), target_tensor]

        if self._neg_samples > 0:
            if self.target_len == 1:
                negs = self._sample_negatives(
                    owner_user, self._neg_samples, exclude_item=target_item
                )
            else:
                negs = self._sample_negatives_rows(
                    owner_user, self._neg_samples, n_rows=self.target_len, exclude_items=target_items
                )
            ret.append(negs)

        return tuple(ret)


@sampler_registry.register()
class UserSequentialSampler(SequentialSampler):
    """`SequentialSampler`, with the owning (private) user index prepended:
    (user, sequence, length, target[, negatives]). Used by sequential models
    that condition on a learned per-user embedding (e.g. Caser) in addition to
    the item sequence itself.

    Args:
        **params (Any): Forwarded to `SequentialSampler.__init__`.
    """

    def sample(self, it: int) -> Tuple[torch.Tensor, ...]:
        """Build the (user, sequence, length, target[, negatives]) sample for
        event `it`.

        Args:
            it (int): Event index.

        Returns:
            Tuple[torch.Tensor, ...]: `super().sample(it)`, with the owning
                user index prepended.
        """
        owner_user = int(self._flat_users[self._valid_target_indices[it]])
        return torch.tensor(owner_user, dtype=torch.long), *super().sample(it)


@sampler_registry.register()
class SameTargetSequentialSampler(PipelineSessionSampler):
    """Next-item prediction paired with a second sequence sampled from a
    different valid target position that shares the same target item.

    Args:
        **params (Any): Forwarded to `PipelineSessionSampler.__init__`.
    """

    def __init__(self, **params: Any):
        super().__init__(**params)

        # Group valid target positions by their target item, for the semantic-positive
        # lookup: one sort instead of one full scan per distinct item. The stable sort
        # keeps each group's positions ascending, which `sample` relies on
        target_items = self._flat_items[self._valid_target_indices]
        order = np.argsort(target_items, kind="stable")
        unique_items, group_starts = np.unique(target_items[order], return_index=True)

        self._target_to_indices: Dict[int, np.ndarray] = dict(zip(
            unique_items.tolist(), np.split(self._valid_target_indices[order], group_starts[1:])
        ))

    def sample(self, it: int) -> Tuple[torch.Tensor, ...]:
        """Build the (sequence, length, target, semantic-positive sequence, length,
        has-semantic-positive) sample for event `it`.

        Args:
            it (int): Event index.

        Returns:
            Tuple[torch.Tensor, ...]: `(seq_tensor, seq_len, pos_item, sem_tensor,
                sem_len, has_semantic_positive)`.
        """
        target_idx = int(self._valid_target_indices[it])
        pos_item = int(self._flat_items[target_idx])

        seq_tensor, seq_len = self._build_padded_sequence(
            target_idx, self._boundary_start_of(target_idx)
        )

        # Find another sequence (not this one) that shares the same target item
        candidates = self._target_to_indices[pos_item]
        n_others = len(candidates) - 1

        if n_others:
            # Draw among the others by skipping this sample's own (sorted) slot, in
            # O(log n) instead of materializing every other candidate
            pick = self._r_int(n_others)
            if pick >= np.searchsorted(candidates, target_idx):
                pick += 1
            sampled_idx = int(candidates[pick])

            sem_tensor, sem_len = self._build_padded_sequence(
                sampled_idx, self._boundary_start_of(sampled_idx)
            )
            has_semantic_positive = True

        # No other sequence shares this target: fall back to a self-copy
        else:
            sem_tensor, sem_len = seq_tensor.clone(), seq_len
            has_semantic_positive = False

        return (
            seq_tensor,
            torch.tensor(seq_len, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            sem_tensor,
            torch.tensor(sem_len, dtype=torch.long),
            torch.tensor(has_semantic_positive, dtype=torch.bool),
        )


@sampler_registry.register()
class SlidingWindowSampler(PipelineSessionSampler):
    """Sequence-to-sequence windows (length `max_seq_len`, step `stride`) within a
    boundary segment, left-aligned and right-padded.

    Windows are anchored at the segment's end and slide backwards in time, so the
    most recent items are always covered: when `stride` doesn't divide the
    segment evenly, only its oldest items are left out. A segment no longer than
    `max_seq_len` yields a single window holding all of it.

    Args:
        stride (int): Step, in flat tape positions, between consecutive windows
            within a boundary segment. Defaults to 1.
        max_windows (int, optional): Keep at most this many windows per segment,
            the most recent ones. Defaults to None, keeping every window.
        **params (Any): Forwarded to `PipelineSessionSampler.__init__`.
    """

    def __init__(self, stride: int = 1, max_windows: Optional[int] = None, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self.stride = stride
        self.max_windows = max_windows

        self._window_starts, self._window_ends = self._compute_windows()
        self.events = len(self._window_starts)

    def _compute_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute every window's `[start, end)` flat tape span.

        Only segments of at least two items host a window: a single item gives
        nothing to predict, same as `_compute_valid_targets`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: `(window_starts, window_ends)`, parallel
                int64 arrays, windows ordered by segment and, within it, oldest first.
        """
        starts, ends = self._segment_spans()
        seg_lens = ends - starts

        # Number of stride-spaced windows that fit in each segment
        counts = np.where(
            seg_lens >= 2, np.maximum(seg_lens - self.max_seq_len, 0) // self.stride + 1, 0
        )
        if self.max_windows is not None:
            counts = np.minimum(counts, self.max_windows)

        # The segment's last window ends at its end, every older one `stride` earlier
        segment, offset = self._ragged_arange(counts)
        window_ends = ends[segment] - (counts[segment] - 1 - offset) * self.stride
        window_starts = np.maximum(starts[segment], window_ends - self.max_seq_len)

        return window_starts, window_ends

    def _window(self, it: int) -> Tuple[torch.Tensor, np.ndarray, int]:
        """Event `it`'s padded window.

        Args:
            it (int): Event index.

        Returns:
            Tuple[torch.Tensor, np.ndarray, int]: `(seq_tensor, seq_items,
                owner_user)`: the padded window, shape `(max_seq_len,)`, its real
                (pre-padding) items as a view of the flat tape, and its owning user.
        """
        start, end = int(self._window_starts[it]), int(self._window_ends[it])
        seq_tensor, _ = self._padded_slice(start, end)
        return seq_tensor, self._flat_items[start:end], int(self._flat_users[start])

    def sample(self, it: int) -> Tuple[torch.Tensor, ...]:
        """Build the (positive sequence[, negative sequence]) window sample for
        event `it`.

        Args:
            it (int): Event index.

        Returns:
            Tuple[torch.Tensor, ...]: `(pos_seq,)`, shape `(max_seq_len,)`, or
                `(pos_seq, neg_seq)` when `self.neg_samples > 0`, `neg_seq` shaped
                `(max_seq_len, neg_samples)` with one negative set per positive
                position (padding positions get padding negatives).
        """
        pos_seq, seq_items, owner_user = self._window(it)

        if self._neg_samples > 0:
            real_len = len(seq_items)
            neg_seq = torch.full(
                (self.max_seq_len, self._neg_samples), self._padding_token, dtype=torch.long
            )

            # One negative set per positive position in the window, all drawn at once
            neg_seq[:real_len] = self._sample_negatives_rows(
                owner_user, self._neg_samples, n_rows=real_len, exclude_items=seq_items
            )
            return pos_seq, neg_seq

        return (pos_seq,)


@sampler_registry.register()
class ClozeSampler(SlidingWindowSampler):
    """BERT4Rec-style masked-language-modeling window sampler.

    Reuses `SlidingWindowSampler`'s windowing: every `stride`-spaced window
    (length `max_seq_len`) within a boundary segment becomes its own masked
    training instance, rather than a single window anchored at the segment's
    end -- so a segment longer than `max_seq_len` contributes more than one
    example per epoch, covering its whole history instead of just its tail.

    Args:
        mask_prob (float): Fraction of a window's items to mask.
        mask_token_id (int): Token id substituted for a masked item.
        stride (int): Step, in flat tape positions, between consecutive windows
            within a boundary segment. Defaults to 1.
        **params (Any): Forwarded to `SlidingWindowSampler.__init__`.
    """

    def __init__(self, mask_prob: float, mask_token_id: int, stride: int = 1, **params: Any):
        super().__init__(stride=stride, **params)

        # Initializing variables
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id

    def sample(self, it: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the (masked sequence, positive targets, negative targets, masked
        positions) sample for event `it`.

        Args:
            it (int): Event index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                `(masked_seq_tensor, pos_items_tensor, neg_items_tensor,
                masked_indices_tensor)`.
        """
        masked_seq_tensor, seq_items, owner_user = self._window(it)
        real_len = len(seq_items)
        window = masked_seq_tensor[:real_len]

        # Randomly mask a fraction of the window's positions
        num_to_mask = max(1, int(real_len * self.mask_prob))
        masked_positions = self._r_choice(real_len, size=num_to_mask, replace=False)

        pos_targets = window[masked_positions]

        # Replace masked positions with the mask token in the input sequence
        window[masked_positions] = self.mask_token_id

        # Pack the pos/neg targets and mask positions into fixed-size tensors
        pos_items_tensor = torch.full(
            (self.max_seq_len,), self._padding_token, dtype=torch.long
        )
        pos_items_tensor[:num_to_mask] = pos_targets

        neg_items_tensor = torch.full(
            (self.max_seq_len, self._neg_samples), self._padding_token, dtype=torch.long
        )
        if self._neg_samples > 0:
            neg_items_tensor[:num_to_mask, :] = self._sample_negatives_rows(
                owner_user, self._neg_samples, n_rows=num_to_mask, exclude_items=pos_targets
            )

        masked_indices_tensor = torch.zeros(self.max_seq_len, dtype=torch.long)
        masked_indices_tensor[:num_to_mask] = masked_positions

        return masked_seq_tensor, pos_items_tensor, neg_items_tensor, masked_indices_tensor
