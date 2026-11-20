from typing import Any, List, Tuple, Union, Sequence

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

        # A user with no rated item, or with no unrated one, can't be used: redraw a usable user
        if not 0 < lui < self._nitems:
            u, ui, lui = self._draw_usable_user()

        i = ui[self._r_int(lui)]
        r = float(self._indexed_ratings[u][i])

        return u, i, r

    def _sample_users(self) -> List[int]:
        """Draw one user per event, uniformly at random with replacement.

        Returns:
            List[int]: The sampled user indices, one per event (a list of plain ints,
                cheaper to index per event than a tensor or an array).
        """
        return self._r_int(0, self._nusers, size=self.events).tolist()


@sampler_registry.register()
class PointWisePosNegRatioRatingsSampler(PipelineSampler):
    """One (user, item, rating) triple per event, item and rating drawn as either a
    positive (with probability `1 / (neg_ratio + 1)`) or a negative (rating 0)
    interaction.

    Args:
        neg_ratio (int): Number of negative draws per positive draw, controlling the
            positive/negative sampling probability.
        implicit (bool): If True, ratings are int labels (1 positive, 0 negative);
            otherwise float, a positive's being its stored value. Defaults to False.
        **params (Any): Forwarded to `PipelineSampler.__init__`.
    """

    def __init__(self, neg_ratio: int, implicit: bool = False, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self.neg_ratio = neg_ratio
        self.implicit = implicit

        self._sampled_users = self._sample_users()

    def sample(self, it: int) -> Sequence[Union[int, float]]:
        """Build the (user, item, rating) triple for event `it`, item and rating
        drawn as either a positive or a negative interaction.

        Args:
            it (int): Event index.

        Returns:
            Sequence[Union[int, float]]: The (user, item, rating) triple.
        """
        u = self._sampled_users[it]
        ui = self._ui_dict[u]
        lui = self._lui_dict[u]

        # A user with no rated item, or with no unrated one, can't be used: redraw a usable user
        if not 0 < lui < self._nitems:
            u, ui, lui = self._draw_usable_user()

        # Positive draw, with probability 1 / (neg_ratio + 1)
        if self._r_int(self.neg_ratio + 1) == 0:
            i = ui[self._r_int(lui)]
            r = 1 if self.implicit else float(self._indexed_ratings[u][i])

        # Negative draw: any item the user hasn't rated
        else:
            i = self._sample_negative(u)
            r = 0 if self.implicit else 0.0

        return u, i, r

    def _sample_users(self) -> List[int]:
        """Draw one user per event, uniformly at random with replacement.

        Returns:
            List[int]: The sampled user indices, one per event (a list of plain ints,
                cheaper to index per event than a tensor or an array).
        """
        return self._r_int(0, self._nusers, size=self.events).tolist()


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
    """One positive plus `m` sampled negative (user, item, label) triples per
    (user, item) training pair.

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
        """Build the positive plus `self.m` negative (user, item, label) triples for
        event `it`.

        Args:
            it (int): Event index.

        Returns:
            Tuple[List[int], List[int], List[int]]: Users, items and labels, each of
                length `m + 1`: the positive (label 1) first, then the negatives
                (label 0).
        """
        u, i = self._pos[it]

        # m negatives for this positive pair, sampled with replacement
        negs = [self._sample_negative(u) for _ in range(self.m)]

        return [u] * (self.m + 1), [i] + negs, [1] + [0] * self.m
