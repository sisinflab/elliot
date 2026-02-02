"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from types import SimpleNamespace
import typing as t
import numpy as np
import logging as pylog
import time

from elliot.recommender.utils import get_model
from elliot.utils import logging

from hyperopt import STATUS_OK
from elliot.hyperoptimization.policy import EvaluationPolicy, FinalPolicy, SearchPolicy


class ModelCoordinator(object):
    """
    This class handles the selection of hyperparameters for the hyperparameter tuning realized with HyperOpt.
    """

    def __init__(self, data_objs, base: SimpleNamespace, params, model_class: t.ClassVar, test_fold_index: int):
        """
        The constructor creates a Placeholder of the recommender model.

        :param base: a SimpleNamespace that contains the configuration (main level) options
        :param params: a SimpleNamespace that contains the hyper-parameters of the model
        :param model_class: the class of the recommendation model
        """
        self.logger = logging.get_logger(self.__class__.__name__, pylog.CRITICAL if base.config_test else pylog.DEBUG)
        self.data_objs = data_objs
        self.base = base
        self.params = params
        self.model_class = model_class
        self.test_fold_index = test_fold_index
        self.model_config_index = 0

    def _build_params(self, args: t.Optional[dict] = None) -> SimpleNamespace:
        base_params = self.params[0] if isinstance(self.params, tuple) else self.params
        model_params = SimpleNamespace(**base_params.__dict__)
        if args:
            for k, v in args.items():
                model_params.__setattr__(k, v)
        return model_params

    def run(self, policy: EvaluationPolicy, args: t.Optional[dict] = None) -> dict:
        include_test = policy.include_test
        model_params = self._build_params(args)

        if args:
            self.logger.info("Hyperparameter tuning exploration:")
            for k, v in args.items():
                self.logger.info(f"Exploration for {k}. Value extracted: {v}")
        else:
            self.logger.info("Hyperparameters:")
            for k, v in model_params.__dict__.items():
                self.logger.info(f"{k} set to {v}")

        internal_losses = []
        results = []
        times = []
        for trainval_index, data_obj in enumerate(self.data_objs):
            if args:
                self.logger.info(f"Exploration: Hyperparameter exploration number {self.model_config_index + 1}")
            self.logger.info(f"Exploration: Test Fold exploration number {self.test_fold_index + 1}")
            self.logger.info(f"Exploration: Train-Validation Fold exploration number {trainval_index + 1}")
            model = get_model(data_obj, self.base, model_params, self.model_class)
            tic = time.perf_counter()
            model.train()
            toc = time.perf_counter()
            times.append(toc - tic)
            internal_losses.append(self._get_internal_loss(model))
            results.append(model.get_results())

        self.model_config_index += 1

        results_mean = self._average_results(results, include_test=include_test) if results else {}
        objective = self._compute_objective(results_mean, internal_losses, model_params)

        payload = {
            "loss": objective["loss"],
            "status": STATUS_OK,
            "params": model.get_params(),
            "val_results": {k: result_dict["val_results"] for k, result_dict in results_mean.items()},
            "val_statistical_results": {k: result_dict["val_statistical_results"] for k, result_dict in model.get_results().items()},
            "time": times,
            "name": model.name,
            "objective": objective["meta"],
        }

        if include_test:
            payload.update({
                "test_results": {k: result_dict["test_results"] for k, result_dict in results_mean.items()},
                "test_statistical_results": {k: result_dict["test_statistical_results"] for k, result_dict in model.get_results().items()},
            })

        if objective["meta"]["target"] == "validation_metric":
            metric = objective["meta"]["metric"]
            k = objective["meta"]["k"]
            if metric is not None and k in results_mean:
                payload["val_metric"] = results_mean[k]["val_results"].get(metric)
                if include_test:
                    payload["test_metric"] = results_mean[k]["test_results"].get(metric)

        return payload

    def objective(self, args):
        """
        This function respect the signature, and the return format required for HyperOpt optimization
        :param args: a Dictionary that contains the new hyper-parameter values that will be used in the current run
        :return: it returns a Dictionary with loss, and status being required by HyperOpt,
        and params, and results being required by the framework
        """
        return self.run(SearchPolicy(), args=args)

    def evaluate(self, args: t.Optional[dict] = None):
        return self.run(FinalPolicy(), args=args)

    def single(self):
        """
        This function respect the signature, and the return format required for HyperOpt optimization
        :param args: a Dictionary that contains the new hyper-parameter values that will be used in the current run
        :return: it returns a Dictionary with loss, and status being required by HyperOpt,
        and params, and results being required by the framework
        """

        return self.run(FinalPolicy(), args=None)

    @staticmethod
    def _average_results(results_list, include_test: bool = True):
        ks = list(results_list[0].keys())
        eval_result_types = ["val_results"]
        if include_test:
            eval_result_types.append("test_results")
        metrics = list(results_list[0][ks[0]]["val_results"].keys())
        return {k: {type_: {metric: np.average([fold_result[k][type_][metric]
                                                for fold_result in results_list])
                            for metric in metrics}
                for type_ in eval_result_types}
                for k in ks}

    def _get_internal_loss(self, model):
        losses = getattr(model, "_losses", None)
        if losses:
            return float(np.min(losses))
        loss = getattr(model, "get_loss", None)
        return float(loss()) if callable(loss) else None

    def _resolve_validation_target(self, model_params: SimpleNamespace):
        cutoff_k = getattr(self.base.evaluation, "cutoffs", [self.base.top_k])
        cutoff_k = cutoff_k if isinstance(cutoff_k, list) else [cutoff_k]
        first_metric = self.base.evaluation.simple_metrics[0] if self.base.evaluation.simple_metrics else None
        default_k = cutoff_k[0]
        raw = getattr(model_params.meta, "validation_metric", None) if hasattr(model_params, "meta") else None
        if not raw:
            return first_metric, default_k
        parts = str(raw).split("@")
        metric = parts[0]
        k = int(parts[1]) if len(parts) > 1 else default_k
        return metric, k

    def _compute_objective(self, results_mean: dict, internal_losses: list, model_params: SimpleNamespace):
        target = None
        if hasattr(model_params, "meta"):
            target = getattr(model_params.meta, "optimization_target", None)
            if not target and getattr(model_params.meta, "optimize_internal_loss", False):
                target = "internal_loss"
        target = target or "validation_metric"

        internal_loss = None
        valid_internal = [l for l in internal_losses if l is not None and np.isfinite(l)]
        if valid_internal:
            internal_loss = float(np.average(valid_internal))

        if target == "internal_loss":
            loss = internal_loss if internal_loss is not None else np.inf
            meta = {
                "target": "internal_loss",
                "metric": None,
                "k": None,
                "direction": "minimize",
                "value": loss if np.isfinite(loss) else None,
            }
            return {"loss": loss, "meta": meta}

        metric, k = self._resolve_validation_target(model_params)
        metric_value = None
        if metric is not None and k in results_mean:
            metric_value = results_mean[k]["val_results"].get(metric)

        minimize_metrics = {"MSE", "RMSE", "MAE"}
        if metric_value is not None and metric.upper() in minimize_metrics:
            loss = float(metric_value)
            direction = "minimize"
        else:
            loss = float(-metric_value) if metric_value is not None else np.inf
            direction = "maximize"

        if not np.isfinite(loss):
            # Fallback to internal loss if validation metric is unavailable
            loss = internal_loss if internal_loss is not None else np.inf
            target = "internal_loss"
            metric = None
            k = None
            direction = "minimize"
            metric_value = loss if np.isfinite(loss) else None

        meta = {
            "target": target,
            "metric": metric,
            "k": k,
            "direction": direction,
            "value": metric_value,
        }
        return {"loss": loss, "meta": meta}

    @staticmethod
    def _std_results(results_list):
        ks = list(results_list[0].keys())
        eval_result_types = ["val_results"]
        metrics = list(results_list[0][ks[0]]["val_results"].keys())
        return {k: {type_: {metric: np.std([fold_result[k][type_][metric] for fold_result in results_list])
                            for metric in metrics}
                    for type_ in eval_result_types}
                for k in ks}
