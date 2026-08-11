from typing import Any, List, Tuple, Union
import torch

from elliot.dataset.samplers.base_sampler import TraditionalSampler, PipelineSampler
from elliot.utils.registry import sampler_registry


@sampler_registry.register()
class CustomPointWiseSparseSampler(PipelineSampler):
    """One observed (user, item, rating) triple per event.

    Args:
        **params (Any): Forwarded to `PipelineSampler.__init__`.
    """

    def __init__(self, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self._sampled_users = self._sample_users()

    def sample(self, it: int) -> Tuple[int, int, float]:
        """Build the observed (user, item, rating) triple for event `it`.

        Args:
            it (int): Event index.

        Returns:
            Tuple[int, int, float]: The (user, item, rating) triple.
        """
        u = self._sampled_users[it]
        ui = self._ui_dict[u]
        lui = self._lui_dict[u]

        # A user who has rated every item can't be used here: redraw a different user
        if lui == self._nitems:
            while u in self._sampled_users:
                u = self._r_int(self._nusers)

        i = ui[self._r_int(lui)]
        r = self._indexed_ratings[u][i]

        return u, i, r

    def _sample_users(self) -> Any:
        """Draw one user per event, uniformly at random with replacement.

        Returns:
            Any: The sampled user indices, one per event.
        """
        return self._r_int(0, self._nusers, size=self.events)


@sampler_registry.register()
class PointWisePosNegRatioRatingsSampler(PipelineSampler):
    """One (user, item, rating) triple per event, item and rating drawn as either a
    positive (with probability `1 / (neg_ratio + 1)`) or a negative (rating 0)
    interaction.

    Args:
        neg_ratio (int): Number of negative draws per positive draw, controlling the
            positive/negative sampling probability.
        implicit (bool): If True, a positive sample's rating is always 1 rather than
            its stored rating value. Defaults to False.
        **params (Any): Forwarded to `PipelineSampler.__init__`.
    """

    def __init__(self, neg_ratio: int, implicit: bool = False, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self.neg_ratio = neg_ratio
        self.implicit = implicit

        self._sampled_users = self._sample_users()

    def sample(self, it: int) -> Tuple[int, int, float]:
        """Build the (user, item, rating) triple for event `it`, item and rating
        drawn as either a positive or a negative interaction.

        Args:
            it (int): Event index.

        Returns:
            Tuple[int, int, float]: The (user, item, rating) triple.
        """
        u = self._sampled_users[it]
        ui = self._ui_dict[u]
        lui = self._lui_dict[u]

        # A user who has rated every item can't be used here: redraw a different user
        if lui == self._nitems:
            while u in self._sampled_users:
                u = self._r_int(self._nusers)

        # Randomly decide positive vs. negative at the configured ratio
        boolean_list = [0] * self.neg_ratio + [1]
        self._r_shuffle(boolean_list)

        # Positive draw
        if boolean_list[0]:
            i = ui[self._r_int(lui)]
            r = self._indexed_ratings[u][i] if not self.implicit else 1

        # Negative draw: any item the user hasn't rated
        else:
            i = self._r_int(self._nitems)
            while i in ui:
                i = self._r_int(self._nitems)
            r = 0

        return u, i, r

    def _sample_users(self) -> Any:
        """Draw one user per event, uniformly at random with replacement.

        Returns:
            Any: The sampled user indices, one per event.
        """
        return self._r_int(0, self._nusers, size=self.events)


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
class MFPointWisePosNegSampler(PipelineSampler):
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

        self._pos = [(u, i, 1) for u, items in self._ui_dict.items() for i in items]

    def sample(self, it: int) -> List[Tuple[int, int, int]]:
        """Build one positive triple plus `self.m` sampled negative triples for
        event `it`.

        Args:
            it (int): Event index.

        Returns:
            List[Tuple[int, int, int]]: The positive triple followed by `self.m`
                negative triples, each `(user, item, label)`.
        """
        pos = self._pos[it]
        u, i, _ = pos
        ui = self._ui_dict[u]

        # Sample m distinct negative items (label 0) for this positive pair
        neg = set()
        for _ in range(self.m):
            j = self._r_int(self._nitems)
            while j in ui:
                j = self._r_int(self._nitems)
            neg.add((u, j, 0))

        return [pos] + list(neg)

    def collate_fn(self, batch: List[List[Tuple[int, int, int]]]) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Flatten the per-event lists of triples into a single batch before
        delegating to `PipelineSampler.collate_fn`.

        Args:
            batch (List[List[Tuple[int, int, int]]]): One list of triples per event.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]: One tensor per tuple position.
        """
        # Flatten the per-event triple lists before the shared collate logic
        concatenated = []
        for lst in batch:
            concatenated.extend(lst)

        return super().collate_fn(concatenated)
