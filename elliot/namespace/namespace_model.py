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

from hyperopt import hp
from yaml import FullLoader as FullLoader
from yaml import load

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
_hyper_max_evals = 'hyper_max_evals'
_hyper_opt_alg = 'hyper_opt_alg'
_data_paths = 'data_paths'
_meta = 'meta'
_random_seed = 'random_seed'
_align_side_with_train = "align_side_with_train"


class NameSpaceModel:
    def __init__(self, config_path, base_folder_path_elliot, base_folder_path_config):
        self.base_namespace = SimpleNamespace()

        self._base_folder_path_elliot = base_folder_path_elliot
        self._base_folder_path_config = base_folder_path_config

        self.config_file = open(config_path)
        self.config = load(self.config_file, Loader=FullLoader)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config[_experiment].get(_gpu, -1))

    @staticmethod
    def _set_path(config_path, local_path):
        if os.path.isabs(local_path):
            return os.path.abspath(local_path)
        else:
            if local_path.startswith((".", "..")) or regexp.search(local_path):
                # return f"{config_path}/{local_path}"
                return os.path.abspath(os.sep.join([config_path, local_path]))
            else:
                # the string is an attribute but not a path
                return local_path

    @staticmethod
    def _safe_set_path(config_path, raw_local_path, dataset_name):
        if isinstance(raw_local_path, str):
            local_path = raw_local_path.format(dataset_name)
            if os.path.isabs(local_path):
                return os.path.abspath(local_path)
            else:
                if local_path.startswith((".", "..")) or regexp.search(local_path):
                    return os.path.abspath(os.sep.join([config_path, local_path]))
                else:
                    # the string is an attribute but not a path
                    return local_path
        else:
            return raw_local_path

    def fill_base(self):

        # for path in self.config[_experiment][_data_paths].keys():
        #     self.config[_experiment][_data_paths][path] = \
        #         self.config[_experiment][_data_paths][path].format(self.config[_experiment][_dataset])
        default_results_recs = os.sep.join(["..", "results", "{0}", "recs"])
        default_results_weights = os.sep.join(["..", "results", "{0}", "weights"])
        default_results_performance = os.sep.join(["..", "results", "{0}", "performance"])
        self.config[_experiment][_recs] = os.path.abspath(self.config[_experiment]\
            .get(_recs, self._set_path(self._base_folder_path_config, default_results_recs))\
            .format(self.config[_experiment][_dataset]))
        self.config[_experiment][_weights] = os.path.abspath(self.config[_experiment]\
            .get(_weights, self._set_path(self._base_folder_path_config, default_results_weights)) \
            .format(self.config[_experiment][_dataset]))
        self.config[_experiment][_performance] = os.path.abspath(self.config[_experiment]\
            .get(_performance, self._set_path(self._base_folder_path_config, default_results_performance)) \
            .format(self.config[_experiment][_dataset]))

        self.config[_experiment][_dataloader] = self.config[_experiment].get(_dataloader, "DataSetLoader")
        self.config[_experiment][_version] = self.config[_experiment].get(_version, __version__)


        manage_directories(self.config[_experiment][_recs], self.config[_experiment][_weights],
                           self.config[_experiment][_performance])

        for p in [_data_config, _weights, _recs, _dataset, _top_k, _performance, _logger_config,
                  _log_folder, _dataloader, _splitting, _prefiltering, _evaluation, _external_models_path,
                  _print_triplets, _config_test, _negative_sampling, _binarize, _random_seed, _align_side_with_train,
                  _version]:
            if p == _data_config:
                side_information = self.config[_experiment][p].get("side_information", None)

                if side_information:
                    if isinstance(side_information, list):
                        side_information = [SimpleNamespace(**{k: self._safe_set_path(self._base_folder_path_config, v, self.config[_experiment][_dataset])
                                                 for k, v in side.items()}) for side in side_information]
                        self.config[_experiment][p].update({k: self._safe_set_path(self._base_folder_path_config, v, self.config[_experiment][_dataset])
                                                            for k, v in self.config[_experiment][p].items()})
                        self.config[_experiment][p]["side_information"] = side_information
                        self.config[_experiment][p][_dataloader] = "DataSetLoader"
                        setattr(self.base_namespace, p, SimpleNamespace(**self.config[_experiment][p]))
                    elif isinstance(side_information, dict):
                        side_information = self.config[_experiment][p].get("side_information", {})
                        side_information.update({k: self._safe_set_path(self._base_folder_path_config, v, self.config[_experiment][_dataset])
                                                 for k, v in side_information.items()})
                        side_information = SimpleNamespace(**side_information)
                        self.config[_experiment][p].update({k: self._safe_set_path(self._base_folder_path_config, v, self.config[_experiment][_dataset])
                                                            for k, v in self.config[_experiment][p].items()})
                        self.config[_experiment][p]["side_information"] = side_information
                        self.config[_experiment][p][_dataloader] = self.config[_experiment][p].get(_dataloader,
                                                                                                   "DataSetLoader")
                        setattr(self.base_namespace, p, SimpleNamespace(**self.config[_experiment][p]))
                    else:
                        raise Exception("Side information is neither a list nor a dict. No other options are allowed.")
                else:
                    self.config[_experiment][p]["side_information"] = []
                    self.config[_experiment][p][_dataloader] = self.config[_experiment][p].get(_dataloader,
                                                                                               "DataSetLoader")
                    self.config[_experiment][p].update(
                        {k: self._safe_set_path(self._base_folder_path_config, v, self.config[_experiment][_dataset])
                         for k, v in self.config[_experiment][p].items()})
                    setattr(self.base_namespace, p, SimpleNamespace(**self.config[_experiment][p]))

            elif p == _splitting and self.config[_experiment].get(p, {}):
                self.config[_experiment][p].update({k: self._safe_set_path(self._base_folder_path_config, v, self.config[_experiment][_dataset])
                                                    for k, v in self.config[_experiment][p].items()})
                test_splitting = self.config[_experiment][p].get("test_splitting", {})
                validation_splitting = self.config[_experiment][p].get("validation_splitting", {})

                if test_splitting:
                    test_splitting = SimpleNamespace(**test_splitting)
                    self.config[_experiment][p]["test_splitting"] = test_splitting

                if validation_splitting:
                    validation_splitting = SimpleNamespace(**validation_splitting)
                    self.config[_experiment][p]["validation_splitting"] = validation_splitting

                setattr(self.base_namespace, p, SimpleNamespace(**self.config[_experiment][p]))
            elif p == _prefiltering and self.config[_experiment].get(p, {}):

                if not isinstance(self.config[_experiment][p], list):
                    self.config[_experiment][p] = [self.config[_experiment][p]]

                preprocessing_strategies = [SimpleNamespace(**strategy) for strategy in self.config[_experiment][p]]
                self.config[_experiment][p] = preprocessing_strategies
                setattr(self.base_namespace, p, self.config[_experiment][p])

            elif p == _negative_sampling and self.config[_experiment].get(p, {}):
                self.config[_experiment][p].update({k: self._safe_set_path(self._base_folder_path_config, v, self.config[_experiment][_dataset])
                                                    for k, v in self.config[_experiment][p].items()})
                self.config[_experiment][p] = SimpleNamespace(**self.config[_experiment][p])
                if getattr(self.config[_experiment][p], 'strategy', '') == 'random':
                    path = os.path.abspath(os.sep.join([self._base_folder_path_config, "..", "data",
                                                         self.config[_experiment][_dataset], "negative.tsv"]))
                    setattr(self.config[_experiment][p], 'file_path', path)
                setattr(self.base_namespace, p, self.config[_experiment][p])
            elif p == _evaluation and self.config[_experiment].get(p, {}):
                complex_metrics = self.config[_experiment][p].get("complex_metrics", {})
                paired_ttest = self.config[_experiment][p].get("paired_ttest", {})
                wilcoxon_test = self.config[_experiment][p].get("wilcoxon_test", {})
                for complex_metric in complex_metrics:
                    complex_metric.update({k: self._safe_set_path(self._base_folder_path_config, v, self.config[_experiment][_dataset])
                                           for k, v in complex_metric.items()})
                    # complex_metric.update({k: self._set_path(self._base_folder_path_config,
                    #                                   v.format(self.config[_experiment][_dataset]))
                    #                 for k, v in complex_metric.items() if isinstance(v, str)})
                self.config[_experiment][p]["complex_metrics"] = complex_metrics
                self.config[_experiment][p]["paired_ttest"] = paired_ttest
                self.config[_experiment][p]["wilcoxon_test"] = wilcoxon_test
                setattr(self.base_namespace, p, SimpleNamespace(**self.config[_experiment][p]))
            elif p == _logger_config:
                if not self.config[_experiment].get(p, False):
                    setattr(self.base_namespace, p, os.path.abspath(os.sep.join([self._base_folder_path_elliot, "config", "logger_config.yml"])))
                else:
                    setattr(self.base_namespace, p,
                            self._safe_set_path(self._base_folder_path_config, self.config[_experiment][p], self.config[_experiment][_dataset]))
                # setattr(self.base_namespace, p, f"{self._base_folder_path_elliot}/config/logger_config.yml")
            elif p == _log_folder:
                if not self.config[_experiment].get(p, False):
                    setattr(self.base_namespace, p, os.path.abspath(os.sep.join([self._base_folder_path_elliot, "..", "log"])))
                else:
                    setattr(self.base_namespace, p,
                            self._safe_set_path(self._base_folder_path_config, self.config[_experiment][p], self.config[_experiment][_dataset]))

                # setattr(self.base_namespace, p, f"{self._base_folder_path_elliot}/../log/")
            elif p == _external_models_path and self.config[_experiment].get(p, False):
                self.config[_experiment][p] = self._safe_set_path(self._base_folder_path_config, self.config[_experiment][p], "")
                setattr(self.base_namespace, p, self.config[_experiment][p])
            elif p == _config_test:
                setattr(self.base_namespace, p, self.config[_experiment].get(p, False))
            elif p == _random_seed:
                setattr(self.base_namespace, p, self.config[_experiment].get(p, 42))
            elif p == _binarize:
                setattr(self.base_namespace, p, self.config[_experiment].get(p, False))
            elif p == _align_side_with_train:
                setattr(self.base_namespace, p, self.config[_experiment].get(p, True))
            else:
                if self.config[_experiment].get(p):
                    setattr(self.base_namespace, p, self.config[_experiment][p])

    def fill_model(self):
        for key in self.config[_experiment][_models]:
            meta_model = self.config[_experiment][_models][key].get(_meta, {})
            model_name_space = SimpleNamespace(**self.config[_experiment][_models][key])
            setattr(model_name_space, _meta, SimpleNamespace(**meta_model))
            if any(isinstance(value, list) for value in self.config[_experiment][_models][key].values()):
                space_list = []
                for k, value in self.config[_experiment][_models][key].items():
                    if isinstance(value, list):
                        valid_functions = ["choice",
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
                            func_ = getattr(hp, value[0].replace(" ","").split("(")[0])
                            val_string = value[0].replace(" ", "").split("(")[1].split(")")[0] \
                                if len(value[0].replace(" ", "").split("(")) > 1 else None
                            val = [literal_eval(val_string) if val_string else None]
                            val.extend([literal_eval(val.replace(" ", "").replace(")", "")) if isinstance(val, str) else
                                        val for val in value[1:]])
                            val = [v for v in val if v is not None]
                            space_list.append((k, func_(k, *val)))
                        elif all(isinstance(item, str) for item in value):
                            space_list.append((k, hp.choice(k, value)))
                        else:
                            space_list.append((k, hp.choice(k, literal_eval(
                                "[" + str(",".join([str(v) for v in value])) + "]")
                                                            )))
                _SPACE = OrderedDict(space_list)
                _estimated_evals = reduce(lambda x, y: x*y, [len(param.pos_args) - 1 for _, param in _SPACE.items()], 1)
                _max_evals = meta_model.get(_hyper_max_evals, _estimated_evals)
                if _max_evals <= 0:
                    raise Exception("Only pure value lists can be used without hyper_max_evals option. Please define hyper_max_evals in model/meta configuration.")
                _opt_alg = ho.parse_algorithms(meta_model.get(_hyper_opt_alg, "grid"))
                yield key, (model_name_space, _SPACE, _max_evals, _opt_alg)
            else:
                if key == "RecommendationFolder":
                    folder_path = getattr(model_name_space, "folder", None)
                    if folder_path:
                        onlyfiles = [f for f in os.listdir(folder_path) if isfile(join(folder_path, f))]
                        for file_ in onlyfiles:
                            local_model_name_space = copy.copy(model_name_space)
                            local_model_name_space.path = os.path.join(folder_path, file_)
                            yield "ProxyRecommender", local_model_name_space
                    else:
                        raise Exception("RecommendationFolder meta-model must expose the folder field.")
                else:
                    yield key, model_name_space
