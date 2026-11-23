from typing import Any, Optional, Tuple
import numpy as np
import torch

from elliot.dataset.samplers.base_sampler import SessionSampler
from elliot.utils.registry import sampler_registry


@sampler_registry.register()
class SequentialSampler(SessionSampler):
    """Next-item prediction: (sequence, length, target[, negatives]).

    One event per valid target position: the target's own prefix (its last
    `max_seq_len` predecessors within the boundary segment) is the input, and the
    model is supervised on that single position.

    Args:
        target_len (int): Number of consecutive future items predicted from a
            single context window. Defaults to 1, which keeps `target` a scalar
            and `negatives` shaped `(neg_samples,)` per sample; `target_len > 1`
            shapes them `(target_len,)` and `(target_len, neg_samples)` instead.
        **params (Any): Forwarded to `SessionSampler.__init__`.
    """

    def __init__(self, target_len: int = 1, **params: Any):
        # Set before `super().__init__`, which computes the valid targets from it
        self.target_len = target_len
        super().__init__(**params)

        self._register_target_events()

    def _compute_valid_targets(self) -> np.ndarray:
        """A flat position is a valid target window start iff it has at least one
        predecessor in its boundary segment and the `target_len` items starting
        there don't run past the segment's end.

        Returns:
            np.ndarray: The valid target flat tape positions.
        """
        return super()._compute_valid_targets(horizon=self.target_len)

    def sample(self, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Build the (sequence, length, target[, negatives]) samples for events `indices`.

        Args:
            indices (torch.Tensor): Event indices.

        Returns:
            Tuple[torch.Tensor, ...]: `(seq, seq_len, target)`, plus a trailing
                `negatives` when `self.neg_samples > 0`, all int64: `seq` shaped
                `(batch, max_seq_len)` and `seq_len` `(batch,)`; `target` `(batch,)`
                and `negatives` `(batch, neg_samples)` when `self.target_len == 1`,
                otherwise `(batch, target_len)` and `(batch, target_len, neg_samples)`,
                covering the `target_len` consecutive items starting at each target.
        """
        targets = self._ev_target[indices]

        # The target(s) are the item(s) at this flat position; context is everything before it
        seq, seq_len = self._padded_batch(self._ev_start[indices], targets)

        if self.target_len == 1:
            target_items = self._tape_items[targets].long()
        else:
            horizon = torch.arange(self.target_len, device=targets.device)
            target_items = self._tape_items[targets[:, None] + horizon].long()

        ret = [seq, seq_len, target_items]

        if self._neg_samples > 0:
            owners = self._tape_users[targets]
            if self.target_len > 1:
                owners = owners.repeat_interleave(self.target_len)
            negs = self._draw_session_negatives(owners, self._neg_samples, target_items.reshape(-1))
            ret.append(negs.view(*target_items.shape, self._neg_samples))

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

    def sample(self, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Build the (user, sequence, length, target[, negatives]) samples for
        events `indices`.

        Args:
            indices (torch.Tensor): Event indices.

        Returns:
            Tuple[torch.Tensor, ...]: `super().sample(indices)`, with the
                owning user indices (int64, shape `(batch,)`) prepended.
        """
        owners = self._tape_users[self._ev_target[indices]].long()
        return owners, *super().sample(indices)


@sampler_registry.register()
class SameTargetSequentialSampler(SessionSampler):
    """Next-item prediction paired with a second sequence sampled from a
    different valid target position that shares the same target item.

    Args:
        **params (Any): Forwarded to `SessionSampler.__init__`.
    """

    def __init__(self, **params: Any):
        super().__init__(**params)

        self._register_target_events()

        # Group the events by their target item, for the semantic-positive lookup.
        # The stable sort keeps each group's events (thus target positions) ascending
        target_items = self._flat_items[self._valid_target_indices]
        order = np.argsort(target_items, kind="stable").astype(np.int64)
        _, group_of_sorted, group_sizes = np.unique(
            target_items[order], return_inverse=True, return_counts=True
        )
        group_of_sorted = group_of_sorted.reshape(-1)
        group_starts = np.cumsum(group_sizes) - group_sizes

        # Per event: where its group starts in `order`, its group's size and its own rank in it
        group_start = np.empty(self.events, dtype=np.int64)
        group_size = np.empty(self.events, dtype=np.int64)
        group_rank = np.empty(self.events, dtype=np.int64)
        group_start[order] = group_starts[group_of_sorted]
        group_size[order] = group_sizes[group_of_sorted]
        group_rank[order] = np.arange(self.events, dtype=np.int64) - group_starts[group_of_sorted]

        self._register_state("_same_target_order", order)
        self._register_state("_same_target_start", group_start)
        self._register_state("_same_target_size", group_size)
        self._register_state("_same_target_rank", group_rank)

    def sample(self, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Build the (sequence, length, target, semantic-positive sequence, length,
        has-semantic-positive) samples for events `indices`.

        Args:
            indices (torch.Tensor): Event indices.

        Returns:
            Tuple[torch.Tensor, ...]: `(seq, seq_len, pos_item, sem_seq, sem_len,
                has_semantic_positive)`: int64 sequences shaped
                `(batch, max_seq_len)`, the rest shaped `(batch,)`, the last bool.
        """
        targets = self._ev_target[indices]
        pos_items = self._tape_items[targets].long()
        seq, seq_len = self._padded_batch(self._ev_start[indices], targets)

        # Draw another event sharing the same target item, skipping this one's own
        # slot in its group; with no other one, fall back to a self-copy
        n_others = self._same_target_size[indices] - 1
        has_semantic_positive = n_others > 0

        pick = self._rand_below(n_others)
        pick += pick >= self._same_target_rank[indices]
        slot = (self._same_target_start[indices] + pick).clamp(max=self.events - 1)
        others = torch.where(has_semantic_positive, self._same_target_order[slot], indices)

        sem_seq, sem_len = self._padded_batch(self._ev_start[others], self._ev_target[others])

        return seq, seq_len, pos_items, sem_seq, sem_len, has_semantic_positive


@sampler_registry.register()
class SlidingWindowSampler(SessionSampler):
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
        **params (Any): Forwarded to `SessionSampler.__init__`.
    """

    def __init__(self, stride: int = 1, max_windows: Optional[int] = None, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self.stride = stride
        self.max_windows = max_windows

        window_starts, window_ends = self._compute_windows()
        self.events = len(window_starts)

        self._register_state("_window_starts", window_starts)
        self._register_state("_window_ends", window_ends)

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

    def _windows(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Events `indices`' padded windows.

        Args:
            indices (torch.Tensor): Event indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: `(seq, real, owners)`:
                the padded windows, int64, shape `(batch, max_seq_len)`, the mask of
                their real (pre-padding) positions and their owning users.
        """
        starts = self._window_starts[indices]
        seq, lengths = self._padded_batch(starts, self._window_ends[indices])
        real = torch.arange(self.max_seq_len, device=seq.device) < lengths[:, None]
        return seq, real, self._tape_users[starts]

    def _negatives_per_position(self, owners: torch.Tensor, exclude: torch.Tensor, keep: torch.Tensor):
        """`neg_samples` negatives per window position, excluding the position's own
        `exclude` item, padding negatives wherever `keep` is False.

        Args:
            owners (torch.Tensor): Owning user per window, shape `(batch,)`.
            exclude (torch.Tensor): Item excluded per position, shape `(batch, max_seq_len)`.
            keep (torch.Tensor): Positions that get real negatives, shape `(batch, max_seq_len)`.

        Returns:
            torch.Tensor: The negatives, int64, shape `(batch, max_seq_len, neg_samples)`.
        """
        # Drawn for every position, padding ones included (excluding the padding
        # token, i.e. nothing): no device sync needed to select the kept ones
        negs = self._draw_session_negatives(
            owners.repeat_interleave(self.max_seq_len), self._neg_samples, exclude.reshape(-1)
        ).view(*exclude.shape, self._neg_samples)
        return torch.where(keep[..., None], negs, self._padding_token)

    def sample(self, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Build the (positive sequence[, negative sequence]) window samples for
        events `indices`.

        Args:
            indices (torch.Tensor): Event indices.

        Returns:
            Tuple[torch.Tensor, ...]: `(pos_seq,)`, shape `(batch, max_seq_len)`, or
                `(pos_seq, neg_seq)` when `self.neg_samples > 0`, `neg_seq` shaped
                `(batch, max_seq_len, neg_samples)` with one negative set per positive
                position (padding positions get padding negatives).
        """
        pos_seq, real, owners = self._windows(indices)

        if self._neg_samples > 0:
            return pos_seq, self._negatives_per_position(owners, pos_seq, real)

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

    def sample(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the (masked sequence, positive targets, negative targets, masked
        positions) samples for events `indices`.

        Each window masks `max(1, floor(real_len * mask_prob))` distinct positions,
        drawn uniformly at random. Targets are indexed by mask slot, not by
        sequence position: slot `s < n_masked` holds the `s`-th masked position
        (in `masked_indices`), its original item (in `pos_items`) and its negatives
        (in `neg_items`); later slots hold padding (and position 0).

        Args:
            indices (torch.Tensor): Event indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                `(masked_seq, pos_items, neg_items, masked_indices)`, int64, shaped
                `(batch, max_seq_len)`, except `neg_items`,
                `(batch, max_seq_len, neg_samples)`.
        """
        seq, real, owners = self._windows(indices)
        real_len = real.sum(dim=1)

        # No window masks more than `max_slots` positions: only those slots are
        # drawn, the rest is padding
        max_slots = min(self.max_seq_len, max(1, int(self.max_seq_len * self.mask_prob)))

        # Randomly mask a fraction of the window's positions
        n_masked = (real_len.to(self._float_dtype) * self.mask_prob).floor().long().clamp(1, max_slots)

        # The start of a random permutation of each window's real positions
        # (padding ones last): its first `n_masked` entries are the masked
        # positions, in random order
        keys = self._rand(tuple(seq.shape)).masked_fill(~real, 2.0)
        order = keys.topk(max_slots, dim=1, largest=False, sorted=True).indices
        slot_used = torch.arange(max_slots, device=seq.device) < n_masked[:, None]

        original = seq.gather(1, order)
        targets = torch.where(slot_used, original, self._padding_token)

        # Replace masked positions with the mask token in the input sequence
        masked_seq = seq.scatter(1, order, torch.where(slot_used, self.mask_token_id, original))

        # Pack the slots into `max_seq_len`-wide tensors, padding the unused ones
        pos_items = torch.full_like(seq, self._padding_token)
        pos_items[:, :max_slots] = targets
        masked_indices = torch.zeros_like(seq)
        masked_indices[:, :max_slots] = torch.where(slot_used, order, 0)

        neg_items = torch.full(
            (*seq.shape, self._neg_samples), self._padding_token, dtype=torch.long, device=seq.device
        )
        if self._neg_samples > 0:
            # Drawn for every slot (excluding the padding token, i.e. nothing, on
            # unused ones): no device sync needed to select the used ones
            negs = self._draw_session_negatives(
                owners.repeat_interleave(max_slots), self._neg_samples, targets.reshape(-1)
            ).view(len(seq), max_slots, self._neg_samples)
            neg_items[:, :max_slots] = torch.where(slot_used[..., None], negs, self._padding_token)

        return masked_seq, pos_items, neg_items, masked_indices
