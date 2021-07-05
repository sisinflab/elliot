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
    - metric: MAD
      clustering_name: Happiness
      clustering_file: /home/cheggynho/Documents/UMUAI2019FatRec/ml-1m-2020-03-08/Clusterings/UsersClusterings/user_clustering_happiness.tsv
    - metric: alpha_ndcg
      alpha: 0.2
    - metric: IELD
      content_file: path
"""
__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from time import time
from types import SimpleNamespace
import logging as pylog
import numpy as np

import elliot.dataset.dataset as ds
from elliot.utils import logging
from . import metrics
from . import popularity_utils
from . import relevance


class Evaluator(object):
    def __init__(self, data: ds.DataSet, params: SimpleNamespace):
        """
        Class to manage all the evaluation methods and operation
        :param data: dataset object
        :param k: top-k evaluation
        """
        self.logger = logging.get_logger(self.__class__.__name__, pylog.CRITICAL if data.config.config_test else
                                         pylog.DEBUG)
        self._data = data
        self._params = params
        self._k = getattr(data.config.evaluation, "cutoffs", [data.config.top_k])
        self._k = self._k if isinstance(self._k, list) else [self._k]
        if any(np.array(self._k) > data.config.top_k):
            raise Exception("Cutoff values must be smaller than recommendation list length (top_k)")
        self._rel_threshold = data.config.evaluation.relevance_threshold
        self._paired_ttest = self._data.config.evaluation.paired_ttest
        self._metrics = metrics.parse_metrics(data.config.evaluation.simple_metrics)
        self._complex_metrics = getattr(data.config.evaluation, "complex_metrics", dict())
        #TODO integrate complex metrics in validation metric (the problem is that usually complex metrics generate a complex name that does not match with the base name when looking for the loss value)
        # if _validation_metric.lower() not in [m.lower()
        #                                       for m in data.config.evaluation.simple_metrics]+[m["metric"].lower()
        #                                                                                        for m in self._complex_metrics]:
        #     raise Exception("Validation metric must be in list of general metrics")
        self._test = data.get_test()

        self._pop = popularity_utils.Popularity(self._data)

        self._evaluation_objects = SimpleNamespace(relevance=relevance.Relevance(self._test, self._rel_threshold),
                                                   pop=self._pop,
                                                   num_items=self._data.num_items,
                                                   data = self._data,
                                                   additional_metrics=self._complex_metrics)
        if data.get_validation():
            self._val = data.get_validation()
            self._val_evaluation_objects = SimpleNamespace(relevance=relevance.Relevance(self._val, self._rel_threshold),
                                                           pop=self._pop,
                                                           num_items=self._data.num_items,
                                                           data = self._data,
                                                           additional_metrics=self._complex_metrics)
        self._needed_recommendations = self._compute_needed_recommendations()

    def eval(self, recommendations):
        """
        Runtime Evaluation of Accuracy Performance (top-k)
        :return:
        """
        result_dict = {}
        for k in self._k:
            val_results, val_statistical_results, test_results, test_statistical_results = self.eval_at_k(recommendations, k)
            local_result_dict ={"val_results": val_results,
                                "val_statistical_results": val_statistical_results,
                                "test_results": test_results,
                                "test_statistical_results": test_statistical_results}
            result_dict[k] = local_result_dict
        return result_dict

    def eval_at_k(self, recommendations, k):
        val_test = ["Validation", "Test"]
        result_list = []
        for p, (test_data, eval_objs) in enumerate(self._get_test_data()):
            if eval_objs is not None:
                eval_objs.cutoff = k
            results, statistical_results = self._process_test_data(recommendations[p], test_data, eval_objs, val_test[p])
            result_list.append((results, statistical_results))

        if (not result_list[0][0]):
            return result_list[1][0], result_list[1][1], result_list[1][0], result_list[1][1]
        elif (not result_list[1][0]):
            return result_list[0][0], result_list[0][1], result_list[0][0], result_list[0][1]
        else:
            return result_list[0][0], result_list[0][1], result_list[1][0], result_list[1][1]

    def _get_test_data(self):
        return [(self._val if hasattr(self, '_val') else None,
                 self._val_evaluation_objects if hasattr(self, '_val_evaluation_objects') else None),
                (self._test if hasattr(self, '_test') else None,
                 self._evaluation_objects if hasattr(self, '_evaluation_objects') else None)
                ]

    def _process_test_data(self, recommendations, test_data, eval_objs, val_test):
        if (not test_data) or (not eval_objs):
            return None, None
        else:
            recommendations = {u: recs for u, recs in recommendations.items() if test_data.get(u, [])}
            rounding_factor = 5
            eval_start_time = time()

            metric_objects = [m(recommendations, self._data.config, self._params, eval_objs) for m in self._metrics]
            for metric in self._complex_metrics:
                metric_objects.extend(metrics.parse_metric(metric["metric"])(recommendations, self._data.config,
                                                                             self._params, eval_objs, metric).get())
            results = {m.name(): m.eval() for m in metric_objects}

            str_results = {k: str(round(v, rounding_factor)) for k, v in results.items()}
            # res_print = "\t".join([":".join(e) for e in str_results.items()])
            self.logger.info("")
            self.logger.info(f"{val_test} Evaluation results")
            self.logger.info(f"Cut-off: {eval_objs.cutoff}")
            self.logger.info(f"Eval Time: {time() - eval_start_time}")
            self.logger.info(f"Results")
            [self.logger.info("\t".join(e)) for e in str_results.items()]

            statistical_results = {}
            if self._paired_ttest:
                statistical_results = {metric_object.name(): metric_object.eval_user_metric()
                                       for metric_object in
                                       [m(recommendations, self._data.config, self._params, eval_objs) for m
                                        in self._metrics]
                                       if isinstance(metric_object, metrics.StatisticalMetric)}
            return results, statistical_results

    def _compute_needed_recommendations(self):
        full_recommendations_metrics = any([m.needs_full_recommendations() for m in self._metrics])
        full_recommendations_additional_metrics = any([metrics.parse_metric(metric["metric"]).needs_full_recommendations() for metric in self._complex_metrics])
        if full_recommendations_metrics:
            self.logger.warn("At least one basic metric requires full length recommendations")
        if full_recommendations_additional_metrics:
            self.logger.warn("At least one additional metric requires full length recommendations", None, 1, None)
        if full_recommendations_metrics or full_recommendations_metrics:
            return self._data.num_items
        else:
            return self._data.config.top_k

    def get_needed_recommendations(self):
        return self._needed_recommendations
