from typing import Any, Tuple
import torch

from elliot.dataset.samplers.base_sampler import InteractionSampler
from elliot.utils.registry import sampler_registry


@sampler_registry.register()
class CustomPointWiseSparseSampler(InteractionSampler):
    """One observed (user, item, rating) triple per event.

    Args:
        **params (Any): Forwarded to `InteractionSampler.__init__`.
    """

    def __init__(self, **params: Any):
        super().__init__(**params)

        self._build_interaction_state(with_ratings=True)

        # Initializing variables
        self._register_state("_sampled_users", self._draw_users())

    def sample(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the observed (user, item, rating) triples for events `indices`.

        Args:
            indices (torch.Tensor): Event indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Users and items (int64)
                and ratings (float32), shape `(batch,)` each.
        """
        # A user with no rated item, or with no unrated one, can't be used: redraw a usable user
        users = self._redraw_unusable_users(self._sampled_users[indices])

        pos = self._draw_rated(users)
        return users, self._ui_indices[pos], self._ui_ratings[pos]


@sampler_registry.register()
class PointWisePosNegRatioRatingsSampler(InteractionSampler):
    """One (user, item, rating) triple per event, item and rating drawn as either a
    positive (with probability `1 / (neg_ratio + 1)`) or a negative (rating 0)
    interaction.

    Args:
        neg_ratio (int): Number of negative draws per positive draw, controlling the
            positive/negative sampling probability.
        implicit (bool): If True, ratings are int labels (1 positive, 0 negative);
            otherwise float, a positive's being its stored value. Defaults to False.
        **params (Any): Forwarded to `InteractionSampler.__init__`.
    """

    def __init__(self, neg_ratio: int, implicit: bool = False, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self.neg_ratio = neg_ratio
        self.implicit = implicit

        self._build_interaction_state(with_ratings=not implicit)

        self._register_state("_sampled_users", self._draw_users())

    def sample(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the (user, item, rating) triples for events `indices`, each item
        and rating drawn as either a positive or a negative interaction.

        Args:
            indices (torch.Tensor): Event indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Users and items (int64)
                and ratings, shape `(batch,)` each: int64 labels if `implicit`,
                float32 ratings otherwise.
        """
        # A user with no rated item, or with no unrated one, can't be used: redraw a usable user
        users = self._redraw_unusable_users(self._sampled_users[indices])

        # Positive draw, with probability 1 / (neg_ratio + 1)
        is_pos = self._randint(self.neg_ratio + 1, tuple(users.shape)) == 0

        # Both draws are made for every row, then picked: no device sync needed
        pos = self._draw_rated(users)
        # Negative draw: any item the user hasn't rated
        neg_items = self._draw_unrated(users, 1)[:, 0]

        items = torch.where(is_pos, self._ui_indices[pos], neg_items)
        if self.implicit:
            ratings = is_pos.long()
        else:
            ratings = torch.where(is_pos, self._ui_ratings[pos], torch.zeros_like(self._ui_ratings[pos]))

        return users, items, ratings


@sampler_registry.register()
class PointWisePosNegRatingsSampler(PointWisePosNegRatioRatingsSampler):
    """`PointWisePosNegRatioRatingsSampler` with a fixed 1:1 positive/negative ratio.

    Args:
        **params (Any): Forwarded to `PointWisePosNegRatioRatingsSampler.__init__`.
    """

    def __init__(self, **params: Any):
        super().__init__(
            neg_ratio=1,
            **params
        )


@sampler_registry.register()
class PointWisePosNegSampler(PointWisePosNegRatioRatingsSampler):
    """`PointWisePosNegRatioRatingsSampler` with a fixed 1:1 positive/negative ratio
    and implicit (always-1) positive ratings.

    Args:
        **params (Any): Forwarded to `PointWisePosNegRatioRatingsSampler.__init__`.
    """

    def __init__(self, **params: Any):
        super().__init__(
            neg_ratio=1,
            implicit=True,
            **params
        )


@sampler_registry.register()
class MFPointWisePosNegSampler(InteractionSampler):
    """One positive plus `m` sampled negative (user, item, label) triples per
    (user, item) training pair.

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
        """Build the positive plus `self.m` negative (user, item, label) triples of
        every event in `indices`, each event's triples consecutive.

        Args:
            indices (torch.Tensor): Event indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Users, items and labels,
                int64, shape `(batch * (m + 1),)` each: per event, the positive
                (label 1) first, then the negatives (label 0).
        """
        users, pos = self._pair_users[indices], self._pair_items[indices]

        # m negatives for this positive pair, sampled with replacement
        negs = self._draw_unrated(users, self.m)

        items = torch.cat((pos[:, None], negs), dim=1).reshape(-1)
        labels = torch.zeros((len(users), self.m + 1), dtype=torch.long, device=users.device)
        labels[:, 0] = 1

        return users.repeat_interleave(self.m + 1), items, labels.reshape(-1)
