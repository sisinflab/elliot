from typing import Any, Dict, Tuple
import torch

from elliot.dataset.samplers.base_sampler import InteractionSampler
from elliot.utils.registry import sampler_registry


@sampler_registry.register()
class PairWiseSampler(InteractionSampler):
    """BPR-style (user, positive item, negative item) triples, one per event,
    with the negative item sampled uniformly excluding the user's positives.

    Each event's user is drawn once, at construction time; its positive and
    negative items are drawn afresh every time the event is sampled.

    Args:
        **params (Any): Forwarded to `InteractionSampler.__init__`.
    """

    def __init__(self, **params: Any):
        super().__init__(**params)

        self._build_interaction_state()

        # Initializing variables
        self._register_state("_freq_users", torch.zeros(self._nusers, dtype=torch.long))
        self._register_state("_freq_items", torch.zeros(self._nitems, dtype=torch.long))

        self._register_state("_sampled_users", self._draw_users())

    def sample(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the (user, positive item, negative item) triples for events `indices`.

        Args:
            indices (torch.Tensor): Event indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Users, positive items
                and negative items, int64, shape `(batch,)` each.
        """
        # A user with no rated item, or with no unrated one, can't be used: redraw a usable user
        users = self._redraw_unusable_users(self._sampled_users[indices])

        pos = self._ui_indices[self._draw_rated(users)]

        # Uniformly sample a negative item, excluding this user's positives
        neg = self._draw_unrated(users, 1)[:, 0]

        self._freq_users += torch.bincount(users, minlength=self._nusers)
        self._freq_items += torch.bincount(torch.cat((pos, neg)), minlength=self._nitems)

        return users, pos, neg

    @property
    def freq_users(self) -> Dict[int, int]:
        """Per-user positive-sample draw counts, accumulated across sampled batches."""
        return dict(enumerate(self._freq_users.tolist()))

    @property
    def freq_items(self) -> Dict[int, int]:
        """Per-item sample draw counts (positive and negative), accumulated across
        sampled batches."""
        return dict(enumerate(self._freq_items.tolist()))


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

    def _draw_users(self) -> torch.Tensor:
        """Draw users batch-by-batch, without replacement within each batch.

        Returns:
            torch.Tensor: The drawn private user indices, int64, one per event.
        """
        users = torch.as_tensor(self._users, dtype=torch.long)

        # Draw each batch's users without replacement, batch by batch
        chunks = []
        for b_start in range(0, self.events, self.b_size):
            current_b_size = min(self.b_size, self.events - b_start)
            chunks.append(users[torch.randperm(len(users), generator=self._init_generator)[:current_b_size]])

        return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.long)


@sampler_registry.register()
class MFPairWiseSampler(InteractionSampler):
    """`m` (user, positive item, negative item) triples per (user, item) training
    pair.

    Args:
        m (int): Number of negatives sampled per positive pair.
        **params (Any): Forwarded to `InteractionSampler.__init__`.
    """

    def __init__(self, m: int, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self.m = m

        self._build_interaction_state()
        self._register_usable_pairs()

        self.events = len(self._pair_users)

    def sample(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the `self.m` (user, positive item, negative item) triples of every
        event in `indices`, each event's triples consecutive.

        Args:
            indices (torch.Tensor): Event indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Users, positive items
                and negative items, int64, shape `(batch * m,)` each.
        """
        users, items = self._pair_users[indices], self._pair_items[indices]

        # m negatives per positive pair, sampled with replacement
        negs = self._draw_unrated(users, self.m)

        return users.repeat_interleave(self.m), items.repeat_interleave(self.m), negs.reshape(-1)
