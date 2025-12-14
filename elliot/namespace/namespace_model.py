"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import copy
import os
import re
from ast import literal_eval
from collections import OrderedDict
from functools import reduce
from os.path import isfile, join
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from hyperopt import hp

import elliot.hyperoptimization as ho
from elliot.utils.folder import manage_directories

regexp = re.compile(r'[\D][\w-]+\.[\w-]+')

_experiment = 'experiment'

_version = 'version'
_data_config = "data_config"
_splitting = "splitting"
_evaluation = "evaluation"
_prefiltering = "prefiltering"
_binarize = "binarize"
_negative_sampling = "negative_sampling"
_dataset = 'dataset'
_dataloader = 'dataloader'
_weights = 'path_output_rec_weight'
_performance = 'path_output_rec_performance'
_logger_config = 'path_logger_config'
_log_folder = 'path_log_folder'
_verbose = 'verbose'
_recs = 'path_output_rec_result'
_top_k = 'top_k'
_config_test = 'config_test'
_print_triplets = 'print_results_as_triplets'
_metrics = 'metrics'
_relevance_threshold = 'relevance_threshold'
_paired_ttest = 'paired_ttest'
_wilcoxon_test = 'wilcoxon_test'
_models = 'models'
_recommender = 'recommender'
_gpu = 'gpu'
_external_models_path = 'external_models_path'
_external_posthoc_path = 'external_posthoc_path'
_hyper_max_evals = 'hyper_max_evals'
_hyper_opt_alg = 'hyper_opt_alg'
_data_paths = 'data_paths'
_meta = 'meta'
_random_seed = 'random_seed'
_align_side_with_train = 'align_side_with_train'
_backend = 'backend'


class PathResolver:
    def __init__(self, base_folder_path_config: str):
        self.base_folder_path_config = base_folder_path_config

    def resolve(self, local_path: str, dataset_name: str = "") -> str:
        if os.path.isabs(local_path):
            return os.path.abspath(local_path)
        if local_path.startswith((".", "..")) or regexp.search(local_path):
            return os.path.abspath(os.sep.join([self.base_folder_path_config, local_path]))
        if dataset_name:
            return local_path.format(dataset_name)
        return local_path

    def resolve_safe(self, value: Any, dataset_name: str = "") -> Any:
        if isinstance(value, str):
            try:
                formatted = value.format(dataset_name)
            except Exception:
                formatted = value
            return self.resolve(formatted, dataset_name)
        if isinstance(value, list):
            return [self.resolve_safe(v, dataset_name) for v in value]
        if isinstance(value, dict):
            return {k: self.resolve_safe(v, dataset_name) for k, v in value.items()}
        return value


class ConfigContext:
    def __init__(self, config: Dict[str, Any], base_folder_path_elliot: str, base_folder_path_config: str):
        self.config = config
        self.experiment = config[_experiment]
        self.base_folder_path_elliot = base_folder_path_elliot
        self.base_folder_path_config = base_folder_path_config
        self.base_namespace = SimpleNamespace()
        self.used_keys = set()
        self.path_resolver = PathResolver(base_folder_path_config)

    def mark_used(self, *keys: str):
        self.used_keys.update(keys)

    def remaining_items(self) -> Dict[str, Any]:
        return {k: v for k, v in self.experiment.items() if k not in self.used_keys}


class NameSpaceModel:
    def __init__(self, config: Dict[str, Any], base_folder_path_elliot: str, base_folder_path_config: str):
        self.context = ConfigContext(config, base_folder_path_elliot, base_folder_path_config)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.context.experiment.get(_gpu, -1))

    @property
    def base_namespace(self) -> SimpleNamespace:
        return self.context.base_namespace

    def fill_base(self):
        processors = [
            self._prepare_output_paths,
            self._build_data_config,
            self._build_splitting,
            self._build_prefiltering,
            self._build_negative_sampling,
            self._build_evaluation,
            self._resolve_logger_config,
            self._resolve_log_folder,
            self._resolve_external_paths,
            self._set_misc_fields,
        ]

        for processor in processors:
            processor()

        # Attach any remaining experiment-level keys directly for flexibility
        for key, value in self.context.remaining_items().items():
            setattr(self.context.base_namespace, key, value)

    def fill_model(self):
        for key, model_config in self.context.experiment[_models].items():
            yield from self._build_model_entry(key, model_config)

    def _prepare_output_paths(self) -> None:
        exp_cfg = self.context.experiment
        dataset_name = exp_cfg[_dataset]
        resolver = self.context.path_resolver
        default_results_recs = os.sep.join(["..", "results", "{0}", "recs"])
        default_results_weights = os.sep.join(["..", "results", "{0}", "weights"])
        default_results_performance = os.sep.join(["..", "results", "{0}", "performance"])

        def resolve_output_path(raw_value: Optional[str], default_value: str) -> str:
            candidate = raw_value if raw_value is not None else default_value
            resolved = resolver.resolve_safe(candidate, dataset_name)
            try:
                resolved = resolved.format(dataset_name)
            except Exception:
                pass
            return os.path.abspath(resolved)

        exp_cfg[_recs] = resolve_output_path(exp_cfg.get(_recs), default_results_recs)
        exp_cfg[_weights] = resolve_output_path(exp_cfg.get(_weights), default_results_weights)
        exp_cfg[_performance] = resolve_output_path(exp_cfg.get(_performance), default_results_performance)

        exp_cfg[_dataloader] = exp_cfg.get(_dataloader, "DataSetLoader")
        exp_cfg[_version] = exp_cfg.get(_version, __version__)

        manage_directories(exp_cfg[_recs], exp_cfg[_weights], exp_cfg[_performance])
        self.context.mark_used(_recs, _weights, _performance, _dataloader, _version)

    def _build_data_config(self) -> None:
        exp_cfg = self.context.experiment
        data_cfg = exp_cfg[_data_config]
        dataset_name = exp_cfg[_dataset]
        resolver = self.context.path_resolver

        side_information = data_cfg.get("side_information", None)
        if side_information:
            if isinstance(side_information, list):
                side_information = [
                    SimpleNamespace(**{k: resolver.resolve_safe(v, dataset_name) for k, v in side.items()})
                    for side in side_information
                ]
                data_cfg.update({k: resolver.resolve_safe(v, dataset_name) for k, v in data_cfg.items()})
                data_cfg["side_information"] = side_information
                data_cfg[_dataloader] = "DataSetLoader"
            elif isinstance(side_information, dict):
                side_information.update({k: resolver.resolve_safe(v, dataset_name) for k, v in side_information.items()})
                side_information = SimpleNamespace(**side_information)
                data_cfg.update({k: resolver.resolve_safe(v, dataset_name) for k, v in data_cfg.items()})
                data_cfg["side_information"] = side_information
                data_cfg[_dataloader] = data_cfg.get(_dataloader, "DataSetLoader")
            else:
                raise Exception("Side information is neither a list nor a dict. No other options are allowed.")
        else:
            data_cfg["side_information"] = []
            data_cfg[_dataloader] = data_cfg.get(_dataloader, "DataSetLoader")
            data_cfg.update({k: resolver.resolve_safe(v, dataset_name) for k, v in data_cfg.items()})

        exp_cfg[_data_config] = data_cfg
        self.context.base_namespace.data_config = SimpleNamespace(**data_cfg)
        self.context.mark_used(_data_config)

    def _build_splitting(self) -> None:
        exp_cfg = self.context.experiment
        splitting_cfg = exp_cfg.get(_splitting, {})
        if not splitting_cfg:
            return

        dataset_name = exp_cfg[_dataset]
        resolver = self.context.path_resolver
        splitting_cfg.update({k: resolver.resolve_safe(v, dataset_name) for k, v in splitting_cfg.items()})
        test_splitting = splitting_cfg.get("test_splitting", {})
        validation_splitting = splitting_cfg.get("validation_splitting", {})

        if test_splitting:
            splitting_cfg["test_splitting"] = SimpleNamespace(**test_splitting)

        if validation_splitting:
            splitting_cfg["validation_splitting"] = SimpleNamespace(**validation_splitting)

        exp_cfg[_splitting] = splitting_cfg
        self.context.base_namespace.splitting = SimpleNamespace(**splitting_cfg)
        self.context.mark_used(_splitting)

    def _build_prefiltering(self) -> None:
        exp_cfg = self.context.experiment
        prefilter_cfg = exp_cfg.get(_prefiltering, {})
        if not prefilter_cfg:
            return

        if not isinstance(prefilter_cfg, list):
            prefilter_cfg = [prefilter_cfg]

        preprocessing_strategies = [SimpleNamespace(**strategy) for strategy in prefilter_cfg]
        exp_cfg[_prefiltering] = preprocessing_strategies
        self.context.base_namespace.prefiltering = preprocessing_strategies
        self.context.mark_used(_prefiltering)

    def _build_negative_sampling(self) -> None:
        exp_cfg = self.context.experiment
        negative_sampling = exp_cfg.get(_negative_sampling, {})
        if not negative_sampling:
            return

        dataset_name = exp_cfg[_dataset]
        resolver = self.context.path_resolver
        negative_sampling.update({k: resolver.resolve_safe(v, dataset_name) for k, v in negative_sampling.items()})
        negative_sampling = SimpleNamespace(**negative_sampling)
        if getattr(negative_sampling, 'strategy', '') == 'random':
            negative_file_path = os.path.abspath(
                os.sep.join([self.context.base_folder_path_config, "..", "data", dataset_name, "negative.tsv"])
            )
            setattr(negative_sampling, 'file_path', negative_file_path)

        exp_cfg[_negative_sampling] = negative_sampling
        self.context.base_namespace.negative_sampling = negative_sampling
        self.context.mark_used(_negative_sampling)

    def _build_evaluation(self) -> None:
        exp_cfg = self.context.experiment
        evaluation_cfg = exp_cfg.get(_evaluation, {})

        dataset_name = exp_cfg[_dataset]
        resolver = self.context.path_resolver
        complex_metrics = evaluation_cfg.get("complex_metrics", {})
        paired_ttest = evaluation_cfg.get("paired_ttest", {})
        wilcoxon_test = evaluation_cfg.get("wilcoxon_test", {})

        for complex_metric in complex_metrics:
            complex_metric.update({k: resolver.resolve_safe(v, dataset_name) for k, v in complex_metric.items()})

        evaluation_cfg["complex_metrics"] = complex_metrics
        evaluation_cfg["paired_ttest"] = paired_ttest
        evaluation_cfg["wilcoxon_test"] = wilcoxon_test

        exp_cfg[_evaluation] = evaluation_cfg
        self.context.base_namespace.evaluation = SimpleNamespace(**evaluation_cfg)
        self.context.mark_used(_evaluation)

    def _resolve_logger_config(self) -> None:
        exp_cfg = self.context.experiment
        resolver = self.context.path_resolver
        if not exp_cfg.get(_logger_config, False):
            path_logger_config = os.path.abspath(
                os.sep.join([self.context.base_folder_path_elliot, "config", "logger_config.yml"])
            )
        else:
            path_logger_config = resolver.resolve_safe(exp_cfg[_logger_config], exp_cfg[_dataset])

        self.context.base_namespace.path_logger_config = path_logger_config
        self.context.mark_used(_logger_config)

    def _resolve_log_folder(self) -> None:
        exp_cfg = self.context.experiment
        resolver = self.context.path_resolver
        if not exp_cfg.get(_log_folder, False):
            path_log_folder = os.path.abspath(os.sep.join([self.context.base_folder_path_elliot, "..", "log"]))
        else:
            path_log_folder = resolver.resolve_safe(exp_cfg[_log_folder], exp_cfg[_dataset])

        self.context.base_namespace.path_log_folder = path_log_folder
        self.context.mark_used(_log_folder)

    def _resolve_external_paths(self) -> None:
        exp_cfg = self.context.experiment
        resolver = self.context.path_resolver
        if exp_cfg.get(_external_models_path, False):
            resolved = resolver.resolve_safe(exp_cfg[_external_models_path], "")
            exp_cfg[_external_models_path] = resolved
            self.context.base_namespace.external_models_path = resolved
            self.context.mark_used(_external_models_path)

        if exp_cfg.get(_external_posthoc_path, False):
            resolved = resolver.resolve_safe(exp_cfg[_external_posthoc_path], "")
            exp_cfg[_external_posthoc_path] = resolved
            self.context.base_namespace.external_posthoc_path = resolved
            self.context.mark_used(_external_posthoc_path)

    def _set_misc_fields(self) -> None:
        exp_cfg = self.context.experiment
        self.context.base_namespace.config_test = exp_cfg.get(_config_test, False)
        self.context.base_namespace.binarize = exp_cfg.get(_binarize, False)
        self.context.base_namespace.random_seed = exp_cfg.get(_random_seed, 42)
        self.context.base_namespace.align_side_with_train = exp_cfg.get(_align_side_with_train, True)
        backend = exp_cfg.get(_backend, ["tensorflow"])
        self.context.base_namespace.backend = backend if isinstance(backend, list) else [backend]

        for key in [_weights, _recs, _performance, _dataset, _top_k, _dataloader, _version, _print_triplets]:
            if exp_cfg.get(key) is not None:
                setattr(self.context.base_namespace, key, exp_cfg.get(key))

        self.context.mark_used(
            _config_test,
            _binarize,
            _random_seed,
            _align_side_with_train,
            _backend,
            _weights,
            _recs,
            _performance,
            _dataset,
            _top_k,
            _dataloader,
            _version,
            _print_triplets,
            _models,
        )

    def _build_model_entry(self, key: str, model_config: Dict[str, Any]):
        meta_model = model_config.get(_meta, {})
        model_name_space = SimpleNamespace(**model_config)
        setattr(model_name_space, _meta, SimpleNamespace(**meta_model))

        if any(isinstance(value, list) for value in model_config.values()):
            space, max_evals, opt_alg = self._build_hyperopt_space(model_config, meta_model)
            yield key, (model_name_space, space, max_evals, opt_alg)
            return

        if key == "RecommendationFolder":
            folder_path = getattr(model_name_space, "folder", None)
            if not folder_path:
                raise Exception("RecommendationFolder meta-model must expose the folder field.")
            onlyfiles = [f for f in os.listdir(folder_path) if isfile(join(folder_path, f))]
            for file_ in onlyfiles:
                local_model_name_space = copy.copy(model_name_space)
                local_model_name_space.path = os.path.join(folder_path, file_)
                yield "ProxyRecommender", local_model_name_space
            return

        yield key, model_name_space

    def _build_hyperopt_space(self, model_config: Dict[str, Any], meta_model: Dict[str, Any]):
        space_list = []
        for k, value in model_config.items():
            if not isinstance(value, list):
                continue
            valid_functions = [
                "choice",
                "randint",
                "uniform",
                "quniform",
                "loguniform",
                "qloguniform",
                "normal",
                "qnormal",
                "lognormal",
                "qlognormal"
            ]
            if isinstance(value[0], str) and value[0] in valid_functions:
                func_ = getattr(hp, value[0].replace(" ", "").split("(")[0])
                val_string = value[0].replace(" ", "").split("(")[1].split(")")[0] \
                    if len(value[0].replace(" ", "").split("(")) > 1 else None
                val_items = [literal_eval(val_string) if val_string else None]
                val_items.extend(
                    [
                        literal_eval(item.replace(" ", "").replace(")", "")) if isinstance(item, str) else item
                        for item in value[1:]
                    ]
                )
                val_items = [v for v in val_items if v is not None]
                space_list.append((k, func_(k, *val_items)))
            elif all(isinstance(item, str) for item in value):
                space_list.append((k, hp.choice(k, value)))
            else:
                space_list.append((k, hp.choice(k, literal_eval("[" + str(",".join([str(v) for v in value])) + "]"))))

        _SPACE = OrderedDict(space_list)
        _estimated_evals = reduce(lambda x, y: x * y, [len(param.pos_args) - 1 for _, param in _SPACE.items()], 1)
        _max_evals = meta_model.get(_hyper_max_evals, _estimated_evals)
        if _max_evals <= 0:
            raise Exception("Only pure value lists can be used without hyper_max_evals option. Please define hyper_max_evals in model/meta configuration.")
        _opt_alg = ho.parse_algorithms(meta_model.get(_hyper_opt_alg, "grid"))
        return _SPACE, _max_evals, _opt_alg
