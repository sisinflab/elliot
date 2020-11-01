import time
from . import metrics
import dataset.dataset as ds

class Evaluator(object):
    def __init__(self, data: ds.DataSet):
        """
        Class to manage all the evaluation methods and operation
        :param data: dataset object
        :param k: top-k evaluation
        """
        self.data = data
        self.k = data.params.k
        # self.model = model
        self.rel_threshold = data.params.rel
        self.metrics = metrics.parse_metrics(data.params.metrics)
        self.test = data.get_test()
        self.relevant_items = self.binary_relevance_filter()

    def binary_relevance_filter(self):
        return {u: [i for i,r in test_items.items() if r >= self.rel_threshold] for u, test_items in self.test.items()}

    def eval(self, recommendations):
        """
        Runtime Evaluation of Accuracy Performance (top-k)
        :return:
        """

        eval_start_time = time()

        results = {m.name(): m(recommendations, self.cutoff, self.relevant_items).eval() for m in self.metrics}

        print(f"Eval Time: {time() - eval_start_time}")

        res_print = "\n".join(["\t".join(e) for e in results.items()])

        print(f"*** Results ***\n{res_print}\n***************")

        return results
