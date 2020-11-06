from time import time
from types import SimpleNamespace

from . import metrics
from . import relevance
import dataset.dataset as ds


class Evaluator(object):
    def __init__(self, data: ds.DataSet):
        """
        Class to manage all the evaluation methods and operation
        :param data: dataset object
        :param k: top-k evaluation
        """
        self._data = data
        self._k = data.config.top_k
        self._rel_threshold = data.config.relevance
        self._metrics = metrics.parse_metrics(data.config.metrics)
        self._test = data.get_test()

        self._evaluation_objects = SimpleNamespace(relevance=relevance.Relevance(self._test, self._rel_threshold))
        # self._data.params.relevant_items = self._binary_relevance_filter()
        # self._data.params.gain_relevance_map = self._compute_user_gain_map()

    # def _compute_user_gain_map(self) -> t.Dict:
    #     """
    #     Method to compute the Gain Map:
    #     rel = 2**(score - threshold + 1) - 1
    #     :param sorted_item_predictions:
    #     :param sorted_item_scores:
    #     :param threshold:
    #     :return:
    #     """
    #     return {u: {i: 0 if score < self._rel_threshold else 2 ** (score - self._rel_threshold + 1) - 1
    #                 for i, score in test_items.items()}
    #             for u, test_items in self._test.items()}
    #
    # def _binary_relevance_filter(self):
    #     """
    #     Binary Relevance filtering for the test items
    #     :return:
    #     """
    #     return {u: [i for i, r in test_items.items() if r >= self._rel_threshold] for u, test_items in self._test.items()}

    def eval(self, recommendations):
        """
        Runtime Evaluation of Accuracy Performance (top-k)
        :return:
        """
        recommendations = {u: recs for u, recs in recommendations.items() if self._test[u]}
        rounding_factor = 5
        eval_start_time = time()

        results = {
            m.name(): m(recommendations, self._data.config, self._data.params, self._evaluation_objects).eval()
            for m in self._metrics
        }
        str_results = {k: str(round(v, rounding_factor)) for k, v in results.items()}
        print(f"\nEval Time: {time() - eval_start_time}")

        res_print = "\n".join(["\t".join(e) for e in str_results.items()])

        print(f"*** Results ***\n{res_print}\n***************\n")

        return results
