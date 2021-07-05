"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Alejandro Bellogín'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, alejandro.bellogin@uam.es'

import typing as t
import math
from abc import ABC, abstractmethod


class Relevance(object):
    def __init__(self, test, rel_threshold):
        self._test = test
        self._rel_threshold = rel_threshold
        self._binary_relevance = None
        self._discounted_relevance = None

    def get_test(self):
        return self._test

    ############## Discounted relevance ##############

    @property
    def discounted_relevance(self):
        if self._discounted_relevance is None:
            self._discounted_relevance = DiscountedRelevance(self._test, self._rel_threshold)
        return self._discounted_relevance

    ############## Binary relevance ##############

    @property
    def binary_relevance(self):
        if self._binary_relevance is None:
            self._binary_relevance = BinaryRelevance(self._test, self._rel_threshold)
        return self._binary_relevance


class AbstractRelevanceSingleton(ABC):

    @abstractmethod
    def get_rel(self, user, item):
        raise NotImplementedError

    @staticmethod
    def logarithmic_ranking_discount(k: int) -> float:
        """
        Method to compute logarithmic discount
        :param k:
        :return:
        """
        return 1 / math.log(k + 2) * math.log(2)


class DiscountedRelevance(AbstractRelevanceSingleton):
    def __init__(self, test, rel_threshold):
        self._discounted_relevance = self._compute_user_gain_map(test, rel_threshold)

    def get_user_rel_gains(self, user):
        return self._discounted_relevance.get(user, {})

    def get_user_rel(self, user):
        return list(self._discounted_relevance.get(user, {}).keys())

    def get_rel(self, user, item):
        return self._discounted_relevance.get(user, {}).get(item, 0)

    def _compute_user_gain_map(self, test, rel_threshold) -> t.Dict:
        """
        Method to compute the Gain Map:
        rel = 2**(score - threshold + 1) - 1
        :param sorted_item_predictions:
        :param sorted_item_scores:
        :param threshold:
        :return:
        """
        return {u: {i: 2 ** (score - rel_threshold + 1) - 1
                    for i, score in test_items.items() if score >= rel_threshold}
                for u, test_items in test.items()}


class BinaryRelevance(AbstractRelevanceSingleton):
    def __init__(self, test, rel_threshold):
        self._binary_relevance = {u: [i for i, r in test_items.items() if r >= rel_threshold] for u, test_items in test.items()}

    def get_user_rel_gains(self, user):
        return dict.fromkeys(self._binary_relevance.get(user, []), 1)

    def get_user_rel(self, user):
        return self._binary_relevance.get(user, [])

    def get_rel(self, user, item):
        return 1 if item in self._binary_relevance.get(user, []) else 0

