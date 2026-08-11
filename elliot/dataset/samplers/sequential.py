from typing import Any, Dict, Tuple
import numpy as np
import torch

from elliot.dataset.samplers.base_sampler import SessionSampler
from elliot.utils.registry import sampler_registry


@sampler_registry.register()
class SequentialSampler(SessionSampler):
    """Next-item prediction: (sequence, length, target[, negatives]).

    Args:
        **params (Any): Forwarded to `SessionSampler.__init__`.
    """

    def __init__(self, **params: Any):
        super().__init__(**params)

    def sample(self, it: int) -> Tuple[torch.Tensor, ...]:
        """Build the (sequence, length, target[, negatives]) sample for event `it`.

        Args:
            it (int): Event index.

        Returns:
            Tuple[torch.Tensor, ...]: `(seq_tensor, seq_len, target_item)`, plus a
                trailing `negatives` tensor when `self.neg_samples > 0`.
        """
        target_idx = int(self._valid_target_indices[it])
        boundary_start = self._boundary_start_of(target_idx)

        # The target itself is the item at this flat position; context is everything before it
        seq_tensor, seq_len = self._build_padded_sequence(target_idx, boundary_start)
        target_item = int(self._flat_items[target_idx])
        owner_user = int(self._flat_users[target_idx])

        ret = [
            seq_tensor,
            torch.tensor(seq_len, dtype=torch.long), torch.tensor(target_item, dtype=torch.long)
        ]

        if self._neg_samples > 0:
            negs = self._sample_negatives(
                owner_user, self._neg_samples, exclude_item=target_item
            )
            ret.append(torch.tensor(negs, dtype=torch.long))

        return tuple(ret)


@sampler_registry.register()
class SameTargetSequentialSampler(SessionSampler):
    """Next-item prediction paired with a second sequence sampled from a
    different boundary segment that shares the same target item.

    Args:
        **params (Any): Forwarded to `SessionSampler.__init__`.
    """

    def __init__(self, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self._target_to_indices: Dict[int, np.ndarray] = {}

        # Group valid target positions by their target item, for the semantic-positive lookup
        target_items = self._flat_items[self._valid_target_indices]
        self._target_to_indices = {
            int(t): self._valid_target_indices[target_items == t]
            for t in np.unique(target_items)
        }

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
        others = candidates[candidates != target_idx]

        if len(others):
            sampled_idx = int(others[self._r_int(len(others))])
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
class SlidingWindowSampler(SessionSampler):
    """Sequence-to-sequence windows (length `max_seq_len`, step `stride`)
    within a boundary segment.

    Args:
        stride (int): Step, in flat tape positions, between consecutive windows
            within a boundary segment. Defaults to 1.
        **params (Any): Forwarded to `SessionSampler.__init__`.
    """

    def __init__(self, stride: int = 1, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self.stride = stride

        self._window_starts, self._window_boundary = self._compute_windows()
        self.events = len(self._window_starts)

    def _compute_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute every window's start position and owning boundary segment id,
        striding over each boundary segment long enough to hold at least two items.

        Returns:
            Tuple[np.ndarray, np.ndarray]: `(window_starts, window_boundary)`, parallel
                arrays giving each window's flat tape start position and owning
                boundary segment id.
        """
        # Only segments long enough to hold at least two items can host a window
        seg_lens = np.diff(self._boundaries)
        valid_segments = np.where(seg_lens >= 2)[0]

        if len(valid_segments) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        valid_lens = seg_lens[valid_segments]
        valid_starts = self._boundaries[valid_segments]

        # Number of stride-spaced windows that fit in each segment
        num_windows = np.floor(
            np.maximum(valid_lens - self.max_seq_len, 0) / self.stride
        ).astype(int) + 1
        total = int(np.sum(num_windows))

        window_segments = np.repeat(valid_segments, num_windows)

        cum = np.zeros(len(valid_segments) + 1, dtype=int)
        cum[1:] = np.cumsum(num_windows)

        # Map each flat window index back to its owning segment and local offset
        indices = np.arange(total)
        segment_block = np.searchsorted(cum, indices, side="right") - 1
        local_window_idx = indices - cum[segment_block]

        window_starts = valid_starts[segment_block] + local_window_idx * self.stride

        return window_starts.astype(np.int64), window_segments.astype(np.int64)

    def sample(self, it: int) -> Tuple[torch.Tensor, ...]:
        """Build the (positive sequence[, negative sequence]) window sample for
        event `it`.

        Args:
            it (int): Event index.

        Returns:
            Tuple[torch.Tensor, ...]: `(pos_seq,)`, or `(pos_seq, neg_seq)` when
                `self.neg_samples > 0`.
        """
        start_idx = int(self._window_starts[it])
        boundary_id = int(self._window_boundary[it])
        boundary_end = int(self._boundaries[boundary_id + 1])

        # Clip the window to its own boundary segment's end
        end_idx = min(start_idx + self.max_seq_len, boundary_end)

        seq_array = self._flat_items[start_idx:end_idx]
        real_len = len(seq_array)
        owner_user = int(self._flat_users[start_idx])

        pos_seq = torch.full(
            (self.max_seq_len,), self._padding_token, dtype=torch.long
        )
        pos_seq[:real_len] = torch.from_numpy(seq_array.copy())

        if self._neg_samples > 0:
            neg_seq = torch.full(
                (self.max_seq_len, self._neg_samples), self._padding_token, dtype=torch.long
            )

            # One negative set per positive position in the window
            for t in range(real_len):
                negs = self._sample_negatives(
                    owner_user, self._neg_samples, exclude_item=int(seq_array[t])
                )
                neg_seq[t, :len(negs)] = torch.tensor(negs, dtype=torch.long)

            return pos_seq, neg_seq

        return (pos_seq,)


@sampler_registry.register()
class ClozeSampler(SessionSampler):
    """BERT4Rec-style masked-language-modeling window anchored at the end of
    a boundary segment.

    Args:
        mask_prob (float): Fraction of a window's items to mask.
        mask_token_id (int): Token id substituted for a masked item.
        **params (Any): Forwarded to `SessionSampler.__init__`.
    """

    def __init__(self, mask_prob: float, mask_token_id: int, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id

        self._window_starts, self._window_ends = self._compute_windows()
        self.events = len(self._window_starts)

    def _compute_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute one window per boundary segment long enough to hold at least two
        items, anchored at the segment's end and clipped to `max_seq_len`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: `(window_starts, window_ends)`, parallel
                arrays giving each window's flat tape start/end positions.
        """
        seg_lens = np.diff(self._boundaries)
        valid_segments = np.where(seg_lens >= 2)[0]

        # Anchor each window at its segment's end, clipped to max_seq_len
        starts = self._boundaries[valid_segments]
        ends = self._boundaries[valid_segments + 1]
        window_starts = np.maximum(starts, ends - self.max_seq_len)

        return window_starts.astype(np.int64), ends.astype(np.int64)

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
        start, end = int(self._window_starts[it]), int(self._window_ends[it])
        owner_user = int(self._flat_users[start])

        seq_array = self._flat_items[start:end].copy()
        real_len = len(seq_array)

        # Randomly mask a fraction of the window's positions
        num_to_mask = max(1, int(real_len * self.mask_prob))
        masked_positions = self._r_choice(real_len, size=num_to_mask, replace=False)

        pos_targets = seq_array[masked_positions]

        # Replace masked positions with the mask token in the input sequence
        seq_array[masked_positions] = self.mask_token_id

        neg_targets = np.full(
            (num_to_mask, self._neg_samples), self._padding_token, dtype=np.int64
        )
        if self._neg_samples > 0:
            for i in range(num_to_mask):
                negs = self._sample_negatives(
                    owner_user, self._neg_samples, exclude_item=int(pos_targets[i])
                )
                neg_targets[i, :len(negs)] = negs

        # Pack the masked sequence, its pos/neg targets, and mask positions into fixed-size tensors
        masked_seq_tensor = torch.full(
            (self.max_seq_len,), self._padding_token, dtype=torch.long
        )
        masked_seq_tensor[:real_len] = torch.from_numpy(seq_array)

        pos_items_tensor = torch.full(
            (self.max_seq_len,), self._padding_token, dtype=torch.long
        )
        pos_items_tensor[:num_to_mask] = torch.from_numpy(pos_targets)

        neg_items_tensor = torch.full(
            (self.max_seq_len, self._neg_samples), self._padding_token, dtype=torch.long
        )
        neg_items_tensor[:num_to_mask, :] = torch.from_numpy(neg_targets)

        masked_indices_tensor = torch.zeros(self.max_seq_len, dtype=torch.long)
        masked_indices_tensor[:num_to_mask] = torch.from_numpy(masked_positions.astype(np.int64))

        return masked_seq_tensor, pos_items_tensor, neg_items_tensor, masked_indices_tensor
