import numpy as np
import torch

from elliot.dataset.samplers.base_sampler import SessionSampler
from elliot.utils.registry import sampler_registry


@sampler_registry.register()
class SequentialSampler(SessionSampler):
    """Next-item prediction: (sequence, length, target[, negatives])."""

    def sample(self, it):
        target_idx = int(self._valid_target_indices[it])
        boundary_start = self._boundary_start_of(target_idx)

        seq_tensor, seq_len = self._build_padded_sequence(target_idx, boundary_start)
        target_item = int(self.flat_items[target_idx])
        owner_user = int(self.flat_users[target_idx])

        ret = [seq_tensor, torch.tensor(seq_len, dtype=torch.long), torch.tensor(target_item, dtype=torch.long)]

        if self.neg_samples > 0:
            negs = self._sample_negatives(owner_user, self.neg_samples, exclude_item=target_item)
            ret.append(torch.tensor(negs, dtype=torch.long))

        return tuple(ret)


@sampler_registry.register()
class SameTargetSequentialSampler(SessionSampler):
    """Next-item prediction paired with a second sequence sampled from a
    different boundary segment that shares the same target item."""

    def __init__(self, **params):
        super().__init__(**params)

        target_items = self.flat_items[self._valid_target_indices]
        self._target_to_indices = {
            int(t): self._valid_target_indices[target_items == t]
            for t in np.unique(target_items)
        }

    def sample(self, it):
        target_idx = int(self._valid_target_indices[it])
        pos_item = int(self.flat_items[target_idx])

        seq_tensor, seq_len = self._build_padded_sequence(target_idx, self._boundary_start_of(target_idx))

        candidates = self._target_to_indices[pos_item]
        others = candidates[candidates != target_idx]

        if len(others):
            sampled_idx = int(others[self._r_int(len(others))])
            sem_tensor, sem_len = self._build_padded_sequence(sampled_idx, self._boundary_start_of(sampled_idx))
            has_semantic_positive = True
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
    within a boundary segment."""

    def __init__(self, stride=1, **params):
        super().__init__(**params)
        self.stride = stride
        self._window_starts, self._window_boundary = self._compute_windows()
        self.events = len(self._window_starts)

    def _compute_windows(self):
        seg_lens = np.diff(self._boundaries)
        valid_segments = np.where(seg_lens >= 2)[0]

        if len(valid_segments) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        valid_lens = seg_lens[valid_segments]
        valid_starts = self._boundaries[valid_segments]

        num_windows = np.floor(np.maximum(valid_lens - self.max_seq_len, 0) / self.stride).astype(int) + 1
        total = int(np.sum(num_windows))

        window_segments = np.repeat(valid_segments, num_windows)

        cum = np.zeros(len(valid_segments) + 1, dtype=int)
        cum[1:] = np.cumsum(num_windows)

        indices = np.arange(total)
        segment_block = np.searchsorted(cum, indices, side="right") - 1
        local_window_idx = indices - cum[segment_block]

        window_starts = valid_starts[segment_block] + local_window_idx * self.stride

        return window_starts.astype(np.int64), window_segments.astype(np.int64)

    def sample(self, it):
        start_idx = int(self._window_starts[it])
        boundary_id = int(self._window_boundary[it])
        boundary_end = int(self._boundaries[boundary_id + 1])
        end_idx = min(start_idx + self.max_seq_len, boundary_end)

        seq_array = self.flat_items[start_idx:end_idx]
        real_len = len(seq_array)
        owner_user = int(self.flat_users[start_idx])

        pos_seq = torch.full((self.max_seq_len,), self.padding_token, dtype=torch.long)
        pos_seq[:real_len] = torch.from_numpy(seq_array.copy())

        if self.neg_samples > 0:
            neg_seq = torch.full((self.max_seq_len, self.neg_samples), self.padding_token, dtype=torch.long)
            for t in range(real_len):
                negs = self._sample_negatives(owner_user, self.neg_samples, exclude_item=int(seq_array[t]))
                neg_seq[t, :len(negs)] = torch.tensor(negs, dtype=torch.long)
            return pos_seq, neg_seq

        return (pos_seq,)


@sampler_registry.register()
class ClozeSampler(SessionSampler):
    """BERT4Rec-style masked-language-modeling window anchored at the end of
    a boundary segment."""

    def __init__(self, mask_prob, mask_token_id, **params):
        super().__init__(**params)
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self._window_starts, self._window_ends = self._compute_windows()
        self.events = len(self._window_starts)

    def _compute_windows(self):
        seg_lens = np.diff(self._boundaries)
        valid_segments = np.where(seg_lens >= 2)[0]

        starts = self._boundaries[valid_segments]
        ends = self._boundaries[valid_segments + 1]
        window_starts = np.maximum(starts, ends - self.max_seq_len)

        return window_starts.astype(np.int64), ends.astype(np.int64)

    def sample(self, it):
        start, end = int(self._window_starts[it]), int(self._window_ends[it])
        owner_user = int(self.flat_users[start])

        seq_array = self.flat_items[start:end].copy()
        real_len = len(seq_array)

        num_to_mask = max(1, int(real_len * self.mask_prob))
        masked_positions = self._r_choice(real_len, size=num_to_mask, replace=False)

        pos_targets = seq_array[masked_positions]
        seq_array[masked_positions] = self.mask_token_id

        neg_targets = np.full((num_to_mask, self.neg_samples), self.padding_token, dtype=np.int64)
        if self.neg_samples > 0:
            for i in range(num_to_mask):
                negs = self._sample_negatives(owner_user, self.neg_samples, exclude_item=int(pos_targets[i]))
                neg_targets[i, :len(negs)] = negs

        masked_seq_tensor = torch.full((self.max_seq_len,), self.padding_token, dtype=torch.long)
        masked_seq_tensor[:real_len] = torch.from_numpy(seq_array)

        pos_items_tensor = torch.full((self.max_seq_len,), self.padding_token, dtype=torch.long)
        pos_items_tensor[:num_to_mask] = torch.from_numpy(pos_targets)

        neg_items_tensor = torch.full((self.max_seq_len, self.neg_samples), self.padding_token, dtype=torch.long)
        neg_items_tensor[:num_to_mask, :] = torch.from_numpy(neg_targets)

        masked_indices_tensor = torch.zeros(self.max_seq_len, dtype=torch.long)
        masked_indices_tensor[:num_to_mask] = torch.from_numpy(masked_positions.astype(np.int64))

        return masked_seq_tensor, pos_items_tensor, neg_items_tensor, masked_indices_tensor
