import time
from . import metrics

class Evaluator(object):
    def __init__(self, model, data, k, rel_threshold):
        """
        Class to manage all the evaluation methods and operation
        :param data: dataset object
        :param k: top-k evaluation
        """
        self.data = data
        self.k = k
        self.model = model
        self.rel_threshold = rel_threshold
        self.metrics = [metrics.Precision]
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
