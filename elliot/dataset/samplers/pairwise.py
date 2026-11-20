from typing import Any, Dict, List, Sequence, Tuple
import numpy as np

from elliot.dataset.samplers.base_sampler import TraditionalSampler, PipelineSampler
from elliot.utils.registry import sampler_registry


@sampler_registry.register()
class PairWiseSampler(PipelineSampler):
    """BPR-style (user, positive item, negative item) triples, one per event,
    with the negative item sampled uniformly excluding the user's positives.

    Args:
        **params (Any): Forwarded to `PipelineSampler.__init__`.
    """

    def __init__(self, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self._freq_users = np.zeros(self._nusers, dtype=np.int64)
        self._freq_items = np.zeros(self._nitems, dtype=np.int64)

        self._sampled_users = self._sample_users()

    def sample(self, it: int) -> Sequence[int]:
        """Build the (user, positive item, negative item) triple for event `it`.

        Args:
            it (int): Event index.

        Returns:
            Sequence[int]: The (user, positive item, negative item) triple.
        """
        u = self._sampled_users[it]
        ui = self._ui_dict[u]
        lui = self._lui_dict[u]

        # A user with no rated item, or with no unrated one, can't be used: redraw a usable user
        if not 0 < lui < self._nitems:
            u, ui, lui = self._draw_usable_user()
        self._freq_users[u] += 1

        i = ui[self._r_int(lui)]
        self._freq_items[i] += 1

        # Uniformly sample a negative item, excluding this user's positives
        j = self._sample_negative(u)
        self._freq_items[j] += 1

        return u, i, j

    def _sample_users(self) -> List[int]:
        """Draw one user per event, uniformly at random with replacement.

        Returns:
            List[int]: The sampled user indices, one per event (a list of plain ints,
                cheaper to index per event than a tensor or an array).
        """
        return self._r_int(0, self._nusers, size=self.events).tolist()

    @property
    def freq_users(self) -> Dict[int, int]:
        """Per-user positive-sample draw counts, accumulated across `sample()` calls."""
        return dict(enumerate(self._freq_users))

    @property
    def freq_items(self) -> Dict[int, int]:
        """Per-item sample draw counts (positive and negative), accumulated across
        `sample()` calls."""
        return dict(enumerate(self._freq_items))


@sampler_registry.register()
class PairWiseBatchSampler(PairWiseSampler):
    """`PairWiseSampler` variant that draws users batch-by-batch without
    replacement within each batch, so a batch never repeats a user (subject to
    `b_size` and dataset size).

    Args:
        b_size (int): Batch size used to chunk the without-replacement user draws.
        **params (Any): Forwarded to `PairWiseSampler.__init__`.
    """

    def __init__(self, b_size: int, **params: Any):
        # Initializing variables
        self.b_size = b_size

        super().__init__(**params)

    def _sample_users(self) -> List[int]:
        """Draw users batch-by-batch, without replacement within each batch.

        Returns:
            List[int]: The sampled user indices, one per event.
        """
        sampled_users = []

        # Draw each batch's users without replacement, batch by batch
        for b_start in range(0, self.events, self.b_size):
            b_stop = min(b_start + self.b_size, self.events)
            current_b_size = b_stop - b_start

            b_idx = self._r_sample(len(self._users), k=current_b_size).tolist()

            sampled_users.extend(self._users[i] for i in b_idx)

        return sampled_users


@sampler_registry.register()
class MFPairWiseSampler(PipelineSampler):
    """`m` (user, positive item, negative item) triples per (user, item) training
    pair.

    Args:
        m (int): Number of negatives sampled per positive pair.
        **params (Any): Forwarded to `PipelineSampler.__init__`.
    """

    def __init__(self, m: int, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self.m = m

        self._pos = self._usable_pairs()

        self.events = len(self._pos)

    def sample(self, it: int) -> Tuple[List[int], List[int], List[int]]:
        """Build the `self.m` (user, positive item, negative item) triples for event `it`.

        Args:
            it (int): Event index.

        Returns:
            Tuple[List[int], List[int], List[int]]: Users, positive items and
                negative items, each of length `m`.
        """
        u, i = self._pos[it]

        # m negatives for this positive pair, sampled with replacement
        negs = [self._sample_negative(u) for _ in range(self.m)]

        return [u] * self.m, [i] * self.m, negs
