"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import typing as t
import math


class Relevance(object):
    def __init__(self, test, rel_threshold):
        self._test = test
        self._rel_threshold = rel_threshold
        self._binary_relevance = []
        self._discounted_relevance = []

    def get_binary_relevance(self):
        if not self._binary_relevance:
            self._binary_relevance = self._binary_relevance_filter()
        return self._binary_relevance

    def get_discounted_relevance(self):
        if not self._discounted_relevance:
            self._discounted_relevance = self._compute_user_gain_map()
        return self._discounted_relevance

    def _compute_user_gain_map(self) -> t.Dict:
        """
        Method to compute the Gain Map:
        rel = 2**(score - threshold + 1) - 1
        :param sorted_item_predictions:
        :param sorted_item_scores:
        :param threshold:
        :return:
        """
        return {u: {i: 2 ** (score - self._rel_threshold + 1) - 1
                    for i, score in test_items.items() if score >= self._rel_threshold}
                for u, test_items in self._test.items()}

    @staticmethod
    def logarithmic_ranking_discount(k: int) -> float:
        """
        Method to compute logarithmic discount
        :param k:
        :return:
        """
        return 1 / math.log(k + 2) * math.log(2)

    def _binary_relevance_filter(self):
        """
        Binary Relevance filtering for the test items
        :return:
        """
        return {u: [i for i, r in test_items.items() if r >= self._rel_threshold] for u, test_items in self._test.items()}

    def get_test(self):
        return self._test
