"""
Module description:

"""


from typing import Optional
import copy
import numpy as np
import logging as pylog
import time
from hyperopt import STATUS_OK

from elliot.namespace import RecommenderConfig
from elliot.hyperoptimization.policy import EvaluationPolicy, FinalPolicy, SearchPolicy
from elliot.result_handler import aggregate_val_folds_results
from elliot.utils import logging, get_trainer, get_model


class ModelCoordinator(object):
    """
    This class handles the selection of hyperparameters for the hyperparameter tuning realized with HyperOpt.
    """

    def __init__(self, data_objs, config, model_config, model_name, test_fold_index: int):
        """
        The constructor creates a Placeholder of the recommender model.

        :param base: a SimpleNamespace that contains the configuration (main level) options
        :param params: a SimpleNamespace that contains the hyper-parameters of the model
        :param model_class: the class of the recommendation model
        """
        self.logger = logging.get_logger(
            self.__class__.__name__, pylog.CRITICAL if config.config_test else pylog.DEBUG
        )
        self.data_objs = data_objs
        self.config = config
        self.model_config = model_config
        self.test_fold_index = test_fold_index
        self.model_config_index = 0

        self._model_class = get_model(model_name, config)
        self._trainer_class = get_trainer(self._model_class)

    def run(self, policy: EvaluationPolicy, args: Optional[dict] = None) -> dict:
        include_test = policy.include_test
        phase = "Test" if include_test else "Exploration"
        model_config = copy.deepcopy(self.model_config)

        if not include_test:
            self.logger.info("Hyperparameter tuning exploration:")

        if args is not None:
            for k, v in args.items():
                v = self._coerce_param(k, v)
                setattr(model_config, k, v)
                self.logger.info(
                    f"{phase} for '{k}'. Value extracted: {v}"
                )

        internal_losses = []
        reports = []

        for trainval_index, data_obj in enumerate(self.data_objs):
            if not include_test and args is not None:
                self.logger.info(f"{phase}: Hyperparameter exploration number {self.model_config_index + 1}")
            self.logger.info(f"{phase}: Test Fold number {self.test_fold_index + 1}")
            self.logger.info(f"{phase}: Train-Validation Fold number {trainval_index + 1}")

            trainer = self._trainer_class(
                data=data_obj,
                config=self.config,
                params=model_config,
                model_class=self._model_class
            )

            tic = time.perf_counter()
            report = trainer.train()
            toc = time.perf_counter()

            report["time"] = toc - tic
            reports.append(report)
            internal_losses.append(self._get_internal_loss(trainer))

        self.model_config_index += 1

        aggregated_results = aggregate_val_folds_results(reports, include_test=include_test)
        aggregated_results["status"] = STATUS_OK

        objective = self._compute_objective(
            aggregated_results.get("val_results", {}),
            internal_losses,
            model_config
        )

        payload = {
            "loss": objective["loss"],
            "objective": objective["meta"],
            **aggregated_results
        }

        if objective["meta"]["target"] == "validation_metric":
            metric = objective["meta"]["metric"]
            k = objective["meta"]["k"]
            if metric is not None and k in aggregated_results.get("val_results", {}):
                payload["val_metric"] = aggregated_results["val_results"][k].get(metric)
                if include_test:
                    payload["test_metric"] = aggregated_results["test_results"][k].get(metric)

        return payload

    def _coerce_param(self, name, value):
        annotations = getattr(self._model_class, "__annotations__", {})
        ann = annotations.get(name)
        if ann is int and isinstance(value, (float, np.floating)):
            if float(value).is_integer():
                return int(value)
        return value

    def objective(self, args):
        """
        This function respect the signature, and the return format required for HyperOpt optimization
        :param args: a Dictionary that contains the new hyper-parameter values that will be used in the current run
        :return: it returns a Dictionary with loss, and status being required by HyperOpt,
        and params, and results being required by the framework
        """
        return self.run(SearchPolicy(), args=args)

    def evaluate(self, args: Optional[dict] = None):
        return self.run(FinalPolicy(), args=args)

    def single(self):
        """
        This function respect the signature, and the return format required for HyperOpt optimization
        :param args: a Dictionary that contains the new hyper-parameter values that will be used in the current run
        :return: it returns a Dictionary with loss, and status being required by HyperOpt,
        and params, and results being required by the framework
        """

        return self.run(FinalPolicy(), args=None)

    def _get_internal_loss(self, model):
        losses = getattr(model, "_losses", None)
        if losses:
            return float(np.min(losses))
        loss = getattr(model, "get_loss", None)
        return float(loss()) if callable(loss) else None

    # def _resolve_validation_target(self, model_config: RecommenderConfig):
    #     cutoff_k = self.config.evaluation.cutoffs or [self.config.top_k]
    #
    #     first_metric = (
    #         self.config.evaluation.simple_metrics[0]
    #         if self.config.evaluation.simple_metrics else None
    #     )
    #
    #     default_k = cutoff_k[0]
    #
    #     raw = model_config.meta.validation_metric
    #
    #     if raw is None:
    #         return first_metric, default_k
    #
    #     parts = str(raw).split("@")
    #     metric = parts[0]
    #     k = int(parts[1]) if len(parts) > 1 else default_k
    #
    #     return metric, k

    def _compute_objective(self, val_results: dict, internal_losses: list, model_config: RecommenderConfig):
        target = model_config.meta.optimization_target

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

        metric = model_config.meta.validation_metric
        k = model_config.meta.validation_k

        # metric_value = None
        # if metric is not None and k in val_results:
        metric_value = val_results[k].get(metric)

        minimize_metrics = {"MSE", "RMSE", "MAE"}
        if metric.upper() in minimize_metrics:
            loss = float(metric_value)
            direction = "minimize"
        else:
            loss = float(-metric_value) #if metric_value is not None else np.inf
            direction = "maximize"

        # if not np.isfinite(loss):
        #     # Fallback to internal loss if validation metric is unavailable
        #     loss = internal_loss if internal_loss is not None else np.inf
        #     target = "internal_loss"
        #     metric = None
        #     k = None
        #     direction = "minimize"
        #     metric_value = loss if np.isfinite(loss) else None

        meta = {
            "target": target,
            "metric": metric,
            "k": k,
            "direction": direction,
            "value": metric_value,
        }
        return {"loss": loss, "meta": meta}

    # @staticmethod
    # def _std_results(results_list):
    #     ks = list(results_list[0].keys())
    #     eval_result_types = ["val_results"]
    #     metrics = list(results_list[0][ks[0]]["val_results"].keys())
    #     return {k: {type_: {metric: np.std([fold_result[k][type_][metric] for fold_result in results_list])
    #                         for metric in metrics}
    #                 for type_ in eval_result_types}
    #             for k in ks}
