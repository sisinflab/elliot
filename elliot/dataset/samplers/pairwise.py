from typing import Any, Dict, List, Tuple
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

    def sample(self, it: int) -> Tuple[int, int, int]:
        """Build the (user, positive item, negative item) triple for event `it`.

        Args:
            it (int): Event index.

        Returns:
            Tuple[int, int, int]: The (user, positive item, negative item) triple.
        """
        u = self._sampled_users[it]
        ui = self._ui_dict[u]
        lui = self._lui_dict[u]

        # A user who has rated every item has no valid negative: redraw a different user
        if lui == self._nitems:
            while u in self._sampled_users:
                u = self._r_int(self._nusers)
        self._freq_users[u] += 1

        i = ui[self._r_int(lui)]
        self._freq_items[i] += 1

        # Uniformly sample a negative item, excluding this user's positives
        j = self._r_int(self._nitems)
        while j in ui:
            j = self._r_int(self._nitems)
        self._freq_items[j] += 1

        return u, i, j

    def _sample_users(self) -> np.ndarray:
        """Draw one user per event, uniformly at random with replacement.

        Returns:
            np.ndarray: The sampled user indices, one per event.
        """
        return self._r_int(0, self._nusers, size=self.events)

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

    def _sample_users(self) -> np.ndarray:
        """Draw users batch-by-batch, without replacement within each batch.

        Returns:
            np.ndarray: The sampled user indices, one per event.
        """
        sampled_users = []

        # Draw each batch's users without replacement, batch by batch
        for b_start in range(0, self.events, self.b_size):
            b_stop = min(b_start + self.b_size, self.events)
            current_b_size = b_stop - b_start

            b_users = self._r_sample(self._users, k=current_b_size)

            sampled_users.extend(b_users)

        return np.array(sampled_users)


@sampler_registry.register()
class MFPairWiseSampler(PipelineSampler):
    """One positive plus `m` sampled negatives per (user, item) training pair,
    flattened into a single batch by `collate_fn`.

    Args:
        m (int): Number of negatives sampled per positive pair.
        **params (Any): Forwarded to `PipelineSampler.__init__`.
    """

    def __init__(self, m: int, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self.m = m

        self._pos = [(u, i) for u, items in self._ui_dict.items() for i in items]

    def sample(self, it: int) -> List[Tuple[int, int, int]]:
        """Build one positive pair plus `self.m` sampled negative pairs for event `it`.

        Args:
            it (int): Event index.

        Returns:
            List[Tuple[int, int, int]]: The positive pair followed by `self.m`
                negative pairs, each `(user, item, negative item)`.
        """
        u, i = self._pos[it]
        ui = self._ui_dict[u]

        # Sample m distinct negatives for this positive pair
        samples = set()
        for _ in range(self.m):
            j = self._r_int(self._nitems)
            while j in ui:
                j = self._r_int(self._nitems)
            samples.add((u, i, j))

        return list(samples)

    def collate_fn(self, batch: List[List[Tuple[int, int, int]]]):
        """Flatten the per-event lists of triples into a single batch before
        delegating to `PipelineSampler.collate_fn`.

        Args:
            batch (List[List[Tuple[int, int, int]]]): One list of triples per event.

        Returns:
            Tuple[torch.Tensor, ...]: One tensor per tuple position.
        """
        # Flatten the per-event triple lists before the shared collate logic
        concatenated = []
        for lst in batch:
            concatenated.extend(lst)

        return super().collate_fn(concatenated)
