"""
Module description:

"""


from typing import Optional
import numpy as np
import logging as pylog
import time
from hyperopt import STATUS_OK

from elliot.namespace import RecommenderConfig
from elliot.result_handler import aggregate_val_folds_results
from elliot.utils import logging, get_trainer, split_metric, wandb_logger
from elliot.utils.registry import model_registry


class ModelCoordinator(object):
    """
    This class handles the selection of hyperparameters for the hyperparameter tuning realized with HyperOpt.
    """

    def __init__(
        self,
        train_val_data,
        main_data,
        config,
        model_config,
        model_name,
        test_fold_index: int
    ):
        """
        The constructor creates a Placeholder of the recommender model.

        :param base: a SimpleNamespace that contains the configuration (main level) options
        :param params: a SimpleNamespace that contains the hyper-parameters of the model
        :param model_class: the class of the recommendation model
        """
        self.logger = logging.get_logger(
            self.__class__.__name__, pylog.CRITICAL if config.config_test else pylog.DEBUG
        )
        self.train_val_data = train_val_data
        self.main_data = main_data
        self.config = config
        self.model_config = model_config
        self.test_fold_index = test_fold_index
        self.model_name = model_name
        self.model_config_index = 0

    def run(self, args: Optional[dict] = None) -> dict:
        phase = "Exploration" if args is not None else "Training"
        model_config = self.model_config.model_copy(deep=True)

        if args is not None:
            self.logger.info("Hyperparameter tuning exploration:")
            for k, v in args.items():
                setattr(model_config, k, v)
                self.logger.info(f"{phase} for '{k}'. Value extracted: {v}")

        internal_losses = []
        reports = []
        fold_trends = []

        save_model = len(self.train_val_data) == 1

        for train_val_index, train_val_obj in enumerate(self.train_val_data):
            if args is not None:
                self.logger.info(f"{phase}: Hyperparameter exploration number {self.model_config_index + 1}")
            self.logger.info(f"{phase}: Test Fold number {self.test_fold_index + 1}")
            self.logger.info(f"{phase}: Train-Validation Fold number {train_val_index + 1}")

            model = model_registry.get(
                name=self.model_name,
                params=model_config,
                interactions=train_val_obj.train_set,
                seed=self.config.random_seed
            )

            trainer = get_trainer(model)(
                config=self.config,
                model=model
            )

            tic = time.perf_counter()
            report = trainer.train(train_val_obj)
            toc = time.perf_counter()

            report["time"] = toc - tic
            reports.append(report)
            internal_losses.append(self._get_internal_loss(trainer))
            fold_trends.append(self._extract_fold_trend(trainer))

        self.model_config_index += 1

        aggregated_results = aggregate_val_folds_results(reports)
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

        if save_model:
            payload["best_model"] = trainer.model

        if objective["meta"]["target"] == "validation_metric":
            metric = objective["meta"]["metric"]
            k = objective["meta"]["k"]
            if metric is not None and k in aggregated_results.get("val_results", {}):
                payload["val_metric"] = aggregated_results["val_results"][k].get(metric)

        if args is not None:
            trend = self._aggregate_fold_trends(fold_trends, objective["meta"])
            if trend:
                payload["trend"] = trend

            wandb_logger.log_hyperopt_trial(
                config=self.config,
                model_name=self.model_name,
                test_fold_index=self.test_fold_index,
                trial_index=self.model_config_index,
                hyperparams=args,
                objective=objective["meta"],
                payload=payload,
            )

        return payload

    # def _coerce_param(self, name, value):
    #     annotations = getattr(self._model_class, "__annotations__", {})
    #     ann = annotations.get(name)
    #     if ann is int and isinstance(value, (float, np.floating)):
    #         if float(value).is_integer():
    #             return int(value)
    #     return value

    def objective(self, args):
        """
        This function respect the signature, and the return format required for HyperOpt optimization
        :param args: a Dictionary that contains the new hyper-parameter values that will be used in the current run
        :return: it returns a Dictionary with loss, and status being required by HyperOpt,
        and params, and results being required by the framework
        """
        return self.run(args)

    def single(self):
        """
        This function respect the signature, and the return format required for HyperOpt optimization
        :param args: a Dictionary that contains the new hyper-parameter values that will be used in the current run
        :return: it returns a Dictionary with loss, and status being required by HyperOpt,
        and params, and results being required by the framework
        """
        return self.run()

    def train(self, args):
        phase = "Training"
        self.logger.info(f"{phase}: Test Fold number {self.test_fold_index + 1}")

        model_config = self.model_config.model_copy(deep=True)

        for k, v in args.items():
            setattr(model_config, k, v)

        model = model_registry.get(
            name=self.model_name,
            params=model_config,
            interactions=self.main_data.train_set,
            seed=self.config.random_seed
        )

        trainer = get_trainer(model)(
            config=self.config,
            model=model
        )

        trainer.train(self.main_data, validate=False)

        return trainer.model

    def _get_internal_loss(self, model):
        losses = getattr(model, "_losses", None)
        if losses:
            return float(np.min(losses))
        loss = getattr(model, "get_loss", None)
        return float(loss()) if callable(loss) else None

    @staticmethod
    def _safe_float(value):
        if value is None:
            return None
        try:
            casted = float(value)
        except (TypeError, ValueError):
            return None
        return casted if np.isfinite(casted) else None

    def _extract_fold_trend(self, trainer) -> dict:
        train_history = list(getattr(trainer, "_epoch_train_history", []) or [])
        validation_history = list(getattr(trainer, "_validation_history", []) or [])
        train_points = []
        validation_points = []

        for point in train_history:
            epoch = point.get("epoch")
            if epoch is None:
                continue
            train_points.append(
                {
                    "epoch": int(epoch),
                    "train_loss": self._safe_float(point.get("train_loss")),
                }
            )

        for point in validation_history:
            epoch = point.get("epoch")
            if epoch is None:
                continue
            validation_points.append(
                {
                    "epoch": int(epoch),
                    "val_metric": self._safe_float(point.get("val_metric")),
                    "val_loss": self._safe_float(point.get("val_loss")),
                }
            )

        return {"train": train_points, "validation": validation_points}

    def _aggregate_fold_trends(self, fold_trends: list, objective_meta: dict) -> dict:
        if not fold_trends:
            return {}

        metric = objective_meta.get("metric")
        k = objective_meta.get("k")
        metric_name = f"{metric}@{k}" if metric is not None and k is not None else None

        bucket = {}
        val_bucket = {}
        for fold_points in fold_trends:
            train_points = fold_points.get("train", []) if isinstance(fold_points, dict) else []
            validation_points = fold_points.get("validation", []) if isinstance(fold_points, dict) else []

            for point in train_points:
                epoch = point.get("epoch")
                if epoch is None:
                    continue
                bucket.setdefault(epoch, {"train_loss": []})
                train_loss = point.get("train_loss")
                if train_loss is not None:
                    bucket[epoch]["train_loss"].append(train_loss)

            for point in validation_points:
                epoch = point.get("epoch")
                if epoch is None:
                    continue
                val_bucket.setdefault(epoch, {"val_metric": [], "val_loss": []})
                val_metric = point.get("val_metric")
                val_loss = point.get("val_loss")
                if val_metric is not None:
                    val_bucket[epoch]["val_metric"].append(val_metric)
                if val_loss is not None:
                    val_bucket[epoch]["val_loss"].append(val_loss)

        if not bucket and not val_bucket:
            return {}

        epochs = sorted(bucket.keys())
        val_epochs = sorted(val_bucket.keys())

        def avg_or_none(values):
            return float(np.mean(values)) if values else None

        train_loss = [avg_or_none(bucket[epoch]["train_loss"]) for epoch in epochs]
        val_metric_values = [avg_or_none(val_bucket[epoch]["val_metric"]) for epoch in val_epochs]
        val_loss_values = [avg_or_none(val_bucket[epoch]["val_loss"]) for epoch in val_epochs]

        trend = {
            "epochs": epochs,
            "train_loss": train_loss,
            "val_epochs": val_epochs,
            "val_metric": val_metric_values,
            "val_loss": val_loss_values,
            "val_metric_name": metric_name,
        }

        return trend

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

        metric, k = split_metric(model_config.meta.validation_metric)

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