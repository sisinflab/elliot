"""
Module description:

"""
"""

evaluation:
  basic_metrics: [nDCG, Precision, Recall, ItemCoverage]
  cutoff: 50
  relevance: 1
  paired_ttest: True
  additional_metrics:
    - name: MAD
      class_file: path
    - name: alpha_ndcg
      alpha: 0.2
    - name: IELD
      content_file: path
"""
__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from time import time
from types import SimpleNamespace

from . import metrics
from . import relevance
import dataset.dataset as ds


class Evaluator(object):
    def __init__(self, data: ds.DataSet, params: SimpleNamespace):
        """
        Class to manage all the evaluation methods and operation
        :param data: dataset object
        :param k: top-k evaluation
        """
        self._data = data
        self._params = params
        self._k = getattr(data.config.evaluation, "cutoff", data.config.top_k)
        self._rel_threshold = data.config.evaluation.relevance
        self._paired_ttest = self._data.config.evaluation.paired_ttest
        self._metrics = metrics.parse_metrics(data.config.evaluation.base_metrics)
        #TODO
        self._additional_metrics = data.config.evaluation.additional_metrics
        self._test = data.get_test()

        self._evaluation_objects = SimpleNamespace(relevance=relevance.Relevance(self._test, self._rel_threshold),
                                                   additional_metrics=self._additional_metrics)
        if data.get_validation():
            self._val = data.get_validation()
            self._val_evaluation_objects = SimpleNamespace(relevance=relevance.Relevance(self._val, self._rel_threshold))

    def eval(self, recommendations):
        """
        Runtime Evaluation of Accuracy Performance (top-k)
        :return:
        """
        result_list = []
        for test_data, eval_objs in self.get_test_data():
            results, statistical_results = self.process_test_data(recommendations, test_data, eval_objs)
            result_list.append((results, statistical_results))

        if (not result_list[0][0]) or (not result_list[0][1]):
            return result_list[1][0], result_list[1][1], result_list[1][0], result_list[1][1]
        elif (not result_list[1][0]) or (not result_list[1][1]):
            return result_list[0][0], result_list[0][1], result_list[0][0], result_list[0][1]
        else:
            return result_list[0][0], result_list[0][1], result_list[1][0], result_list[1][1]

    def get_test_data(self):
        return [(self._val if hasattr(self, '_val') else None,
                 self._val_evaluation_objects if hasattr(self, '_val_evaluation_objects') else None),
                (self._test if hasattr(self, '_test') else None,
                 self._evaluation_objects if hasattr(self, '_evaluation_objects') else None)
                ]

    def process_test_data(self, recommendations, test_data, eval_objs):
        if (not test_data) or (not eval_objs):
            return None, None
        else:
            recommendations = {u: recs for u, recs in recommendations.items() if test_data[u]}
            rounding_factor = 5
            eval_start_time = time()

            metric_objects = [m(recommendations, self._data.config, self._params, eval_objs) for m in self._metrics]
            for metric in self._additional_metrics:
                metric_objects.extend(metrics.parse_metric(metric.name)(recommendations, self._data.config, self._params, eval_objs).get())
            results = {m.name(): m.eval() for m in metric_objects}

            str_results = {k: str(round(v, rounding_factor)) for k, v in results.items()}
            print(f"\nEval Time: {time() - eval_start_time}")

            res_print = "\n".join(["\t".join(e) for e in str_results.items()])

            print(f"*** Results ***\n{res_print}\n***************\n")

            statistical_results = {}
            if self._paired_ttest:
                statistical_results = {metric_object.name(): metric_object.eval_user_metric()
                                       for metric_object in
                                       [m(recommendations, self._data.config, self._params, eval_objs) for m
                                        in self._metrics]
                                       if isinstance(metric_object, metrics.StatisticalMetric)}
            return results, statistical_results
