"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from elliot.dataset.samplers.base_sampler import TraditionalSampler, PipelineSampler


class CustomPointWiseSparseSampler(PipelineSampler):
    def __init__(self, **params):
        super().__init__(**params)

        self._sampled_users = self._sample_users()

    def sample(self, it):
        u = self._sampled_users[it]
        ui = self._ui_dict[u]
        lui = self._lui_dict[u]

        if lui == self._nitems:
            while u in self._sampled_users:
                u = self._r_int(self._nusers)

        i = ui[self._r_int(lui)]
        r = self._indexed_ratings[u][i]

        return u, i, r

    def _sample_users(self):
        return self._r_int(0, self._nusers, size=self.events)


class PointWisePosNegRatioRatingsSampler(PipelineSampler):
    def __init__(self, neg_ratio, implicit=False, **params):
        super().__init__(**params)

        self._sampled_users = self._sample_users()

        self.neg_ratio = neg_ratio
        self.implicit = implicit

    def sample(self, it):
        u = self._sampled_users[it]
        ui = self._ui_dict[u]
        lui = self._lui_dict[u]

        if lui == self._nitems:
            while u in self._sampled_users:
                u = self._r_int(self._nusers)

        boolean_list = [0] * self.neg_ratio + [1]
        self._r_shuffle(boolean_list)

        if boolean_list[0]:
            i = ui[self._r_int(lui)]
            r = self._indexed_ratings[u][i] if not self.implicit else 1
        else:
            i = self._r_int(self._nitems)
            while i in ui:
                i = self._r_int(self._nitems)
            r = 0

        return u, i, r

    def _sample_users(self):
        return self._r_int(0, self._nusers, size=self.events)


class PointWisePosNegRatingsSampler(PointWisePosNegRatioRatingsSampler):
    def __init__(self, **params):
        super().__init__(
            neg_ratio=1,
            **params
        )


class PointWisePosNegSampler(PointWisePosNegRatioRatingsSampler):
    def __init__(self, **params):
        super().__init__(
            neg_ratio=1,
            implicit=True,
            **params
        )


class MFPointWisePosNegSampler(PipelineSampler):
    def __init__(self, m, **params):
        super().__init__(**params)

        self._pos = [(u, i, 1) for u, items in self._ui_dict.items() for i in items]
        self.m = m

    def sample(self, it):
        pos = self._pos[it]
        u, i, _ = pos
        ui = self._ui_dict[u]

        neg = set()
        for _ in range(self.m):
            j = self._r_int(self._nitems)
            while j in ui:
                j = self._r_int(self._nitems)
            neg.add((u, j, 0))

        return [pos] + list(neg)

    def collate_fn(self, batch):
        concatenated = []
        for lst in batch:
            concatenated.extend(lst)

        return super().collate_fn(concatenated)
