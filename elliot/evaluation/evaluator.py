from time import time

from . import metrics
import dataset.dataset as ds

class Evaluator(object):
    def __init__(self, data: ds.DataSet):
        """
        Class to manage all the evaluation methods and operation
        :param data: dataset object
        :param k: top-k evaluation
        """
        self._data = data
        self._k = data.params.k
        self._rel_threshold = data.params.rel
        self._metrics = metrics.parse_metrics(data.params.metrics)
        self._test = data.get_test()
        self._data.params.relevant_items = self._binary_relevance_filter()

    def _binary_relevance_filter(self):
        """
        Binary Relevance filtering for the test items
        :return:
        """
        return {u: [i for i, r in test_items.items() if r >= self._rel_threshold] for u, test_items in self._test.items()}

    def eval(self, recommendations):
        """
        Runtime Evaluation of Accuracy Performance (top-k)
        :return:
        """
        recommendations = {u: recs for u, recs in recommendations.items() if self._test[u]}
        rounding_factor = 5
        eval_start_time = time()

        results = {
            m.name(): str(round(m(recommendations, self._data.params).eval(), rounding_factor))
            for m in self._metrics
        }

        print(f"Eval Time: {time() - eval_start_time}")

        res_print = "\n".join(["\t".join(e) for e in results.items()])

        print(f"*** Results ***\n{res_print}\n***************")

        return results
