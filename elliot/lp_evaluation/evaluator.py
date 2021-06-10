"""
Module description:

"""
"""


"""
__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from time import time
from types import SimpleNamespace
import logging as pylog
import numpy as np
import typing as t

from elliot.utils import logging
from . import metrics


class LPEvaluator(object):
    def __init__(self, side: SimpleNamespace, config: SimpleNamespace, params: SimpleNamespace):
        """
        Class to manage all the evaluation methods and operation
        :param data: dataset object
        :param k: top-k evaluation
        """
        self.logger = logging.get_logger(self.__class__.__name__, pylog.DEBUG)
        self._side = side
        self._params = params

        if not (hasattr(self._side, "test_triples") and hasattr(self._side, "all_triples") and hasattr(self._side, "entity_to_idx") and hasattr(self._side, "predicate_to_idx")):
            raise Exception("Side information loader is not compatible with Link prediction evaluation")

        self._metrics = metrics.parse_metrics(config.lp_evaluation.simple_metrics)
        self._batch_size = getattr(config.lp_evaluation, "batch_size", -1)
        self._k = getattr(config.lp_evaluation, "cutoffs", [config.top_k])
        self._k = self._k if isinstance(self._k, list) else [self._k]
        # self._k = [1, 3, 5, 10, 20, 50, 100]

        self._evaluation_objects = self.create_evaluation_object(self._side.all_triples, self._side.entity_to_idx, self._side.predicate_to_idx, self._side.test_triples, "test")

        self._val_evaluation_objects = self.create_evaluation_object(self._side.all_triples, self._side.entity_to_idx, self._side.predicate_to_idx, self._side.dev_triples, "val")

    def create_evaluation_object(self, all_triples, entity_to_index, predicate_to_index, eval_triples, type):
        eval_obj_ns = SimpleNamespace()
        if self._batch_size == -1:
            eval_obj_ns.batch_size = eval_triples.shape[0]
        else:
            eval_obj_ns.batch_size = self._batch_size
        eval_obj_ns.type = type
        eval_obj_ns.test_triples: t.List[t.Tuple[str, str, str]] = eval_triples
        eval_obj_ns.all_triples: t.List[t.Tuple[str, str, str]] = all_triples
        eval_obj_ns.entity_to_index: t.Dict[str, int] = entity_to_index
        eval_obj_ns.predicate_to_index: t.Dict[str, int] = predicate_to_index

        eval_obj_ns.xs = np.array(
            [eval_obj_ns.entity_to_index.get(s) for (s, _, _) in eval_obj_ns.test_triples])
        eval_obj_ns.xp = np.array(
            [eval_obj_ns.predicate_to_index.get(p) for (_, p, _) in eval_obj_ns.test_triples])
        eval_obj_ns.xo = np.array(
            [eval_obj_ns.entity_to_index.get(o) for (_, _, o) in eval_obj_ns.test_triples])

        eval_obj_ns.sp_to_o, eval_obj_ns.po_to_s = {}, {}
        for s, p, o in eval_obj_ns.all_triples:
            s_idx, p_idx, o_idx = eval_obj_ns.entity_to_index.get(
                s), eval_obj_ns.predicate_to_index.get(p), eval_obj_ns.entity_to_index.get(o)
            sp_key = (s_idx, p_idx)
            po_key = (p_idx, o_idx)

            if sp_key not in eval_obj_ns.sp_to_o:
                eval_obj_ns.sp_to_o[sp_key] = []
            if po_key not in eval_obj_ns.po_to_s:
                eval_obj_ns.po_to_s[po_key] = []

            eval_obj_ns.sp_to_o[sp_key] += [o_idx]
            eval_obj_ns.po_to_s[po_key] += [s_idx]

        assert eval_obj_ns.xs.shape == eval_obj_ns.xp.shape == eval_obj_ns.xo.shape
        eval_obj_ns.nb_test_triples = eval_obj_ns.xs.shape[0]

        return eval_obj_ns

    def eval(self, model):
        test_results = self.process_evaluation(model, self._evaluation_objects)
        val_results = self.process_evaluation(model, self._val_evaluation_objects)

        result_dict = {}
        for k in self._k:
            result_dict[k] = {**test_results[k], **val_results[k]}
        return result_dict

    def process_evaluation(self, model, eval_obj):

        ranks_l, ranks_r = [], []

        for start_idx in range(0, eval_obj.nb_test_triples, eval_obj.batch_size):
            end_idx = min(start_idx + eval_obj.batch_size, eval_obj.nb_test_triples)

            batch_xs = eval_obj.xs[start_idx:end_idx]
            batch_xp = eval_obj.xp[start_idx:end_idx]
            batch_xo = eval_obj.xo[start_idx:end_idx]

            scores_sp = model.predict(batch_xp, batch_xs, None).numpy()
            scores_po = model.predict(batch_xp, None, batch_xo).numpy()

            batch_size = batch_xs.shape[0]
            for elem_idx in range(batch_size):
                s_idx, p_idx, o_idx = batch_xs[elem_idx], batch_xp[elem_idx], batch_xo[elem_idx]

                # Code for the filtered setting
                sp_key = (s_idx, p_idx)
                po_key = (p_idx, o_idx)

                o_to_remove = eval_obj.sp_to_o[sp_key]
                s_to_remove = eval_obj.po_to_s[po_key]

                for tmp_o_idx in o_to_remove:
                    if tmp_o_idx != o_idx:
                        scores_sp[elem_idx, tmp_o_idx] = - np.infty

                for tmp_s_idx in s_to_remove:
                    if tmp_s_idx != s_idx:
                        scores_po[elem_idx, tmp_s_idx] = - np.infty
                # End of code for the filtered setting

                rank_l = 1 + np.argsort(np.argsort(- scores_po[elem_idx, :]))[s_idx]
                rank_r = 1 + np.argsort(np.argsort(- scores_sp[elem_idx, :]))[o_idx]

                ranks_l += [rank_l]
                ranks_r += [rank_r]

        result_dict = {}
        for k in self._k:
            result_dict[k] = {f"{eval_obj.type}_results": self.eval_at_k(ranks_l, ranks_r, eval_obj, k),
                              f"{eval_obj.type}_statistical_results": {}}

        return result_dict

    def eval_at_k(self, ranks_l, ranks_r, evaluation_object, k):
        rounding_factor = 5
        eval_start_time = time()

        evaluation_object.cutoff = k
        metric_objects = [m(ranks_l, ranks_r, None, self._params, evaluation_object) for m in self._metrics]
        results = {m.name(): m.eval() for m in metric_objects}
        # print(f"top-{k}\t{results}")

        str_results = {k: str(round(v, rounding_factor)) for k, v in results.items()}
        # res_print = "\t".join([":".join(e) for e in str_results.items()])
        print("")
        print(f"{evaluation_object.type} Evaluation results")
        print(f"Cut-off: {evaluation_object.cutoff}")
        print(f"Eval Time: {time() - eval_start_time}")
        print(f"Results")
        [print("\t".join(e)) for e in str_results.items()]

        return results

    # def _process_test_data(self, recommendations, test_data, eval_objs, val_test):
    #     if (not test_data) or (not eval_objs):
    #         return None, None
    #     else:
    #         recommendations = {u: recs for u, recs in recommendations.items() if test_data.get(u, [])}
    #         rounding_factor = 5
    #         eval_start_time = time()
    #
    #         metric_objects = [m(recommendations, self._data.config, self._params, eval_objs) for m in self._metrics]
    #         for metric in self._complex_metrics:
    #             metric_objects.extend(metrics.parse_metric(metric["metric"])(recommendations, self._data.config,
    #                                                                          self._params, eval_objs, metric).get())
    #         results = {m.name(): m.eval() for m in metric_objects}
    #
    #         str_results = {k: str(round(v, rounding_factor)) for k, v in results.items()}
    #         # res_print = "\t".join([":".join(e) for e in str_results.items()])
    #         self.logger.info("")
    #         self.logger.info(f"{val_test} Evaluation results")
    #         self.logger.info(f"Cut-off: {eval_objs.cutoff}")
    #         self.logger.info(f"Eval Time: {time() - eval_start_time}")
    #         self.logger.info(f"Results")
    #         [self.logger.info("\t".join(e)) for e in str_results.items()]
    #
    #         statistical_results = {}
    #         if self._paired_ttest:
    #             statistical_results = {metric_object.name(): metric_object.eval_user_metric()
    #                                    for metric_object in
    #                                    [m(recommendations, self._data.config, self._params, eval_objs) for m
    #                                     in self._metrics]
    #                                    if isinstance(metric_object, metrics.StatisticalMetric)}
    #         return results, statistical_results

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
