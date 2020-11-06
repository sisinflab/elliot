import typing as t


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
        return {u: {i: 0 if score < self._rel_threshold else 2 ** (score - self._rel_threshold + 1) - 1
                    for i, score in test_items.items()}
                for u, test_items in self._test.items()}

    def _binary_relevance_filter(self):
        """
        Binary Relevance filtering for the test items
        :return:
        """
        return {u: [i for i, r in test_items.items() if r >= self._rel_threshold] for u, test_items in self._test.items()}
