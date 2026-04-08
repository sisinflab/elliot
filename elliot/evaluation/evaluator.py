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

from time import time
from types import SimpleNamespace
import logging as pylog
import math
import numpy as np
from sklearn.metrics import mean_squared_error

from elliot.dataset import DataSet
from elliot.evaluation.metrics import StatisticalMetric
from elliot.evaluation.popularity_utils import Popularity
from elliot.evaluation.relevance import Relevance
from elliot.evaluation.accelerated_metrics import compute_accelerated_metrics, is_supported_metric
from elliot.evaluation.metrics.base_metric import BaseMetric
from elliot.namespace import RecommenderConfig, ExperimentConfig, EvaluationConfig
from elliot.utils import logging, get_device
from elliot.utils.folder import path_absolute, check_path
from elliot.utils.registry import metric_registry



class Evaluator(object):
    eval_config: EvaluationConfig

    def __init__(self, config: ExperimentConfig, params: RecommenderConfig):
        """
        Class to manage all the evaluation methods and operation
        :param data: dataset object
        :param k: top-k evaluation
        """
        self.logger = logging.get_logger(
            self.__class__.__name__, pylog.CRITICAL if config.config_test else pylog.DEBUG
        )
        self._config = config
        self.eval_config = config.evaluation
        self._params = params

        self._k = self.eval_config.cutoffs
        if any(np.array(self._k) > config.top_k):
            raise Exception("Cutoff values must be smaller than recommendation list length (top_k)")

        self._rel_threshold = self.eval_config.relevance_threshold
        self._paired_ttest = self.eval_config.paired_ttest
        self._metrics = self.eval_config.simple_metrics
        self._complex_metrics = self.eval_config.complex_metrics

        selected_device = str(get_device())
        configured_accelerate = self.eval_config.accelerate
        self._accelerate = (
            selected_device in {"cuda", "mps"}
            if configured_accelerate is None
            else bool(configured_accelerate)
        )
        self._accelerate_verify = self.eval_config.accelerate_verify
        self._accelerate_verify_once = self.eval_config.accelerate_verify_once
        self._accelerate_tolerance = self.eval_config.accelerate_tolerance
        self._accelerate_device = str(self.eval_config.accelerate_device or selected_device)
        self._accelerate_verified = False

        #TODO integrate complex metrics in validation metric (the problem is that usually complex metrics generate a complex name that does not match with the base name when looking for the loss value)
        # if _validation_metric.lower() not in [m.lower()
        #                                       for m in data.config.evaluation.simple_metrics]+[m["metric"].lower()
        #                                                                                        for m in self._complex_metrics]:
        #     raise Exception("Validation metric must be in list of general metrics")

        self._needed_recommendations = self._compute_needed_recommendations()

        self._initialized = False

    def _init_state(self, dataset):
        self._data = dataset
        self._eval_users = self._load_eval_users()
        self._pop = Popularity(dataset)
        self._pop_cache = self._build_popularity_cache()
        self._initialized = True

    def eval(self, recommendations, dataset: DataSet, label="test"):
        if not self._initialized:
            self._init_state(dataset)

        if dataset is None:
            raise ValueError("Argument `dataset` cannot be None")

        eval_data = self._apply_eval_user_filter(self._data.eval_set.get_dict())
        if self._eval_users is not None:
            eval_data = self._apply_eval_user_filter(eval_data)

        eval_obj = self._build_eval_object(eval_data)

        result_dict = {}
        for k in self._k:
            if eval_obj is not None:
                eval_obj.cutoff = k
            results, statistical_results = self._process_eval_data(
                recommendations, eval_data, eval_obj, label
            )
            result_dict[k] = {
                f"{label}_results": results,
                f"{label}_statistical_results": statistical_results,
            }

        return result_dict

    def eval_error(self, eval_pred, eval_true):
        """
        Runtime Evaluation of Error-based Performance
        :return:
        """
        eval_results = mean_squared_error(eval_true, eval_pred)
        result_dict = {0: {"results": {'MSE': eval_results}, "statistical_results": []}}
        return result_dict

    def _build_popularity_cache(self):
        train_dict = self._data.train_set.get_dict()
        item_count = {}
        for user_hist in train_dict.values():
            for item in user_hist.keys():
                item_count[item] = item_count.get(item, 0) + 1

        num_users = len(train_dict)
        item_novelty_epc = {}
        item_novelty_efd = {}
        max_nov_efd = 0.0

        if item_count:
            if num_users > 0:
                item_novelty_epc = {item: 1.0 - (count / num_users) for item, count in item_count.items()}
            norm = float(sum(item_count.values()))
            if norm > 0:
                min_count = min(item_count.values())
                max_nov_efd = -math.log2(min_count / norm)
                item_novelty_efd = {
                    item: -math.log2(count / norm)
                    for item, count in item_count.items()
                }

        short_head = set(self._pop.get_short_head())
        long_tail = set(self._pop.get_long_tail())

        return SimpleNamespace(
            item_count=item_count,
            item_novelty_epc=item_novelty_epc,
            item_novelty_efd=item_novelty_efd,
            max_nov_efd=max_nov_efd,
            short_head_set=short_head,
            long_tail_set=long_tail,
            pop_items=self._pop.get_pop_items(),
        )

    def _build_eval_object(self, eval_data):
        if not eval_data:
            return None
        return SimpleNamespace(
            relevance=Relevance(eval_data, self._rel_threshold),
            pop=self._pop,
            pop_cache=self._pop_cache,
            num_items=self._data.train_set.dims[1],
            train_data=self._data.train_set,
            additional_metrics=self._complex_metrics,
        )

    def _load_eval_users(self):
        file_path = self.eval_config.user_filter_file
        if not file_path:
            return None

        file_path = path_absolute(file_path)
        if not check_path(file_path):
            raise FileNotFoundError(f"User filter file not found: {file_path}")

        sep = self.eval_config.reader.sep
        id_space = self.eval_config.user_filter_id_space.strip().lower()

        raw_users = []
        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                token = line
                if sep:
                    token = line.split(sep)[0]
                elif "\t" in line:
                    token = line.split("\t")[0]
                elif "," in line:
                    token = line.split(",")[0]
                else:
                    token = line.split()[0]
                token = token.strip()
                if token:
                    raw_users.append(token)

        if not raw_users:
            self.logger.warning("User filter file is empty; skipping user filtering.")
            return None

        if id_space in {"private", "internal"}:
            users, _ = self._data.get_inverse_mappings()
            mapped = set()
            invalid = 0
            for token in raw_users:
                try:
                    idx = int(token)
                except (TypeError, ValueError):
                    invalid += 1
                    continue
                if 0 <= idx < len(users):
                    mapped.add(users[idx])
                else:
                    invalid += 1
            if invalid:
                self.logger.warning(
                    "Dropped invalid private user ids from filter list",
                    extra={"context": {"count": invalid}}
                )
            return mapped if mapped else None

        casted = set()
        invalid = 0
        for token in raw_users:
            try:
                casted.add(self._cast_user_id(token, str))
            except (TypeError, ValueError):
                invalid += 1
        if invalid:
            self.logger.warning(
                "Dropped invalid public user ids from filter list",
                extra={"context": {"count": invalid}}
            )

        if not casted:
            self.logger.warning("No valid users found in user filter file.")
            return None

        known_users, _ = self._data.get_users_items()
        known_set = set(known_users) if known_users else None
        if known_set is not None:
            filtered = casted & known_set
            dropped = len(casted) - len(filtered)
            if dropped:
                self.logger.warning(
                    "Dropped unknown users from filter list",
                    extra={"context": {"count": dropped}}
                )
            return filtered if filtered else None

        return casted

    def _apply_eval_user_filter(self, eval_dict):
        if not self._eval_users:
            return eval_dict
        filtered = {u: items for u, items in eval_dict.items() if u in self._eval_users}
        self.logger.info(
            "Evaluation user filter applied",
            extra={"context": {"users": len(self._eval_users)}}
        )
        return filtered

    @staticmethod
    def _cast_user_id(token, target_type):
        if target_type in (int, np.int32, np.int64) or np.issubdtype(target_type, np.integer):
            return int(float(token))
        if target_type in (float, np.float32, np.float64) or np.issubdtype(target_type, np.floating):
            return float(token)
        return str(token)

    def _process_eval_data(self, recommendations, eval_data, eval_obj, label):
        if (not eval_data) or (not eval_obj):
            return None, None

        recommendations = {u: recs for u, recs in recommendations.items() if eval_data.get(u, [])}
        rounding_factor = 5
        eval_start_time = time()

        results = {}
        statistical_results = {}

        remaining_metric_names = list(self._metrics)
        accelerated_metric_names = []

        if self._accelerate:
            accelerated_metric_names = [
                metric_name for metric_name in self._metrics
                if is_supported_metric(metric_name)
            ]
            remaining_metric_names = [
                metric_name for metric_name in self._metrics
                if metric_name not in set(accelerated_metric_names)
            ]

            if accelerated_metric_names:
                try:
                    accel = compute_accelerated_metrics(
                        recommendations=recommendations,
                        test_data=eval_data,
                        cutoff=eval_obj.cutoff,
                        relevance_threshold=self._rel_threshold,
                        metric_names=accelerated_metric_names,
                        device=self._accelerate_device or str(get_device()),
                        return_user_metrics=bool(self._paired_ttest),
                    )

                    results.update(accel.results)
                    if self._paired_ttest:
                        statistical_results.update(accel.user_results)

                    self.logger.info(
                        "Accelerated simple metrics enabled",
                        extra={
                            "context": {
                                "phase": label,
                                "device": accel.device,
                                "users": accel.users,
                                "metrics": accelerated_metric_names,
                            }
                        }
                    )

                except Exception as ex:
                    accelerated_metric_names = []
                    remaining_metric_names = list(self._metrics)
                    self.logger.warning(
                        "Accelerated metric evaluation failed, falling back to legacy pipeline",
                        extra={"context": {"error": str(ex)}}
                    )

        legacy_metric_objects = [
            metric_registry.get(
                name=metric_name,
                recommendations=recommendations,
                config=self._config,
                params=self._params,
                eval_objects=eval_obj,
            )
            for metric_name in remaining_metric_names
        ]

        for metric_object in legacy_metric_objects:
            metric_name = metric_object.name
            user_metric = None

            if self._paired_ttest and isinstance(metric_object, StatisticalMetric):
                user_metric = metric_object.eval_user_metric()
                statistical_results[metric_name] = user_metric

            if user_metric is not None and metric_object.__class__.eval is BaseMetric.eval:
                metric_values = list(user_metric.values())
                results[metric_name] = float(np.average(metric_values)) if metric_values else float("nan")
            else:
                results[metric_name] = metric_object.eval()

        if accelerated_metric_names and self._should_verify_accelerated():
            self._verify_accelerated_results(
                accelerated_metric_names=accelerated_metric_names,
                recommendations=recommendations,
                eval_obj=eval_obj,
                scalar_results=results,
                user_results=statistical_results,
            )

        metric_objects = legacy_metric_objects[:]
        for metric in self._complex_metrics:
            metric_objects.extend(
                metric_registry.get(
                    name=metric["metric"],
                    recommendations=recommendations,
                    config=self._config,
                    params=self._params,
                    eval_objects=eval_obj,
                    additional_data=metric,
                ).get()
            )
        for metric_obj in metric_objects:
            if metric_obj.name not in results:
                results[metric_obj.name] = metric_obj.eval()

        str_results = {k: str(round(v, rounding_factor)) for k, v in results.items()}
        # res_print = "\t".join([":".join(e) for e in str_results.items()])
        self.logger.info("")
        self.logger.info(f"{label} Evaluation results")
        self.logger.info(f"Cut-off: {eval_obj.cutoff}")
        self.logger.info(f"Eval Time: {time() - eval_start_time}")
        self.logger.info(f"Results")
        [self.logger.info("\t".join(e)) for e in str_results.items()]

        return results, statistical_results

    def _should_verify_accelerated(self) -> bool:
        if not self._accelerate_verify:
            return False
        if self._accelerate_verify_once and self._accelerate_verified:
            return False
        return True

    def _verify_accelerated_results(
        self,
        accelerated_metric_names,
        recommendations,
        eval_obj,
        scalar_results,
        user_results,
    ):
        if not accelerated_metric_names:
            return

        verify_objects = [
            metric_registry.get(
                name=metric_name,
                recommendations=recommendations,
                config=self._config,
                params=self._params,
                eval_objects=eval_obj,
            )
            for metric_name in self._metrics
            if metric_name in set(accelerated_metric_names)
        ]

        mismatch_found = False
        for metric_object in verify_objects:
            metric_name = metric_object.name
            legacy_value = metric_object.eval()
            accelerated_value = scalar_results.get(metric_name)

            if not self._is_close(accelerated_value, legacy_value):
                mismatch_found = True
                self.logger.error(
                    "Accelerated metric mismatch, using legacy value",
                    extra={"context": {
                        "metric": metric_name,
                        "accelerated": accelerated_value,
                        "legacy": legacy_value,
                        "tolerance": self._accelerate_tolerance,
                    }}
                )
                scalar_results[metric_name] = legacy_value

            if self._paired_ttest and isinstance(metric_object, StatisticalMetric):
                legacy_user = metric_object.eval_user_metric()
                accelerated_user = user_results.get(metric_name, {})
                if not self._is_user_metric_close(accelerated_user, legacy_user):
                    mismatch_found = True
                    self.logger.error(
                        "Accelerated user-wise metric mismatch, using legacy values",
                        extra={"context": {"metric": metric_name}}
                    )
                    user_results[metric_name] = legacy_user

        self._accelerate_verified = not mismatch_found

    def _is_user_metric_close(self, left, right) -> bool:
        if set(left.keys()) != set(right.keys()):
            return False
        return all(self._is_close(left[u], right[u]) for u in left.keys())

    def _is_close(self, left, right) -> bool:
        if left is None or right is None:
            return False
        if isinstance(left, np.generic):
            left = left.item()
        if isinstance(right, np.generic):
            right = right.item()
        try:
            if np.isnan(left) and np.isnan(right):
                return True
        except TypeError:
            return False
        return math.isclose(
            float(left),
            float(right),
            rel_tol=self._accelerate_tolerance,
            abs_tol=self._accelerate_tolerance
        )

    def _compute_needed_recommendations(self):
        full_recs_metrics = False
        for metric_name in self._metrics:
            if metric_registry.get_class(metric_name).needs_full_recommendations:
                self.logger.warning(
                    f"Metric {metric_name} requires full length recommendations"
                )
                full_recs_metrics = True

        full_recs_additional_metrics = False
        for metric in self._complex_metrics:
            if metric_registry.get_class(metric["metric"]).needs_full_recommendations:
                self.logger.warning(
                    f"Additional metric {metric['metric']} requires full length recommendations"
                )
                full_recs_additional_metrics = True

        if full_recs_metrics or full_recs_additional_metrics:
            return self._data.train_set.dims[1]
        else:
            return self._config.top_k

    def get_needed_recommendations(self):
        return self._needed_recommendations
