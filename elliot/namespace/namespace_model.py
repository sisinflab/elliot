"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import os
import sys
from types import SimpleNamespace
from ast import literal_eval
from functools import reduce

from hyperopt import hp
from yaml import FullLoader as FullLoader
from yaml import load
from collections import OrderedDict
from elliot.utils.folder import manage_directories
import elliot.hyperoptimization as ho
import re

regexp = re.compile(r'[\D][\w-]+\.[\w-]+')

_experiment = 'experiment'

_data_config = "data_config"
_splitting = "splitting"
_evaluation = "evaluation"
_prefiltering = "prefiltering"
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


class NameSpaceModel:
    def __init__(self, config_path, base_folder_path_elliot, base_folder_path_config):
        self.base_namespace = SimpleNamespace()

        self._base_folder_path_elliot = base_folder_path_elliot
        self._base_folder_path_config = base_folder_path_config

        self.config_file = open(config_path)
        self.config = load(self.config_file, Loader=FullLoader)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config[_experiment].get(_gpu,-1))

    @staticmethod
    def _set_path(config_path, local_path):
        if os.path.isabs(local_path):
            return local_path
        else:
            if local_path.startswith(("./", "../")) or regexp.search(local_path):
                return f"{config_path}/{local_path}"
            else:
                return local_path

    def fill_base(self):

        # for path in self.config[_experiment][_data_paths].keys():
        #     self.config[_experiment][_data_paths][path] = \
        #         self.config[_experiment][_data_paths][path].format(self.config[_experiment][_dataset])

        self.config[_experiment][_recs] = self.config[_experiment]\
            .get(_recs, self._set_path(self._base_folder_path_config, "../results/{0}/recs/"))\
            .format(self.config[_experiment][_dataset])
        self.config[_experiment][_weights] = self.config[_experiment]\
            .get(_weights, self._set_path(self._base_folder_path_config, "../results/{0}/weights/")) \
            .format(self.config[_experiment][_dataset])
        self.config[_experiment][_performance] = self.config[_experiment]\
            .get(_performance, self._set_path(self._base_folder_path_config, "../results/{0}/performance/")) \
            .format(self.config[_experiment][_dataset])

        self.config[_experiment][_dataloader] = self.config[_experiment].get(_dataloader, "DataSetLoader")

        manage_directories(self.config[_experiment][_recs], self.config[_experiment][_weights],
                           self.config[_experiment][_performance])

        for p in [_data_config, _weights, _recs, _dataset, _top_k, _performance, _logger_config,
                  _log_folder, _dataloader, _splitting, _prefiltering, _evaluation, _external_models_path,
                  _print_triplets, _config_test]:
            if p == _data_config:
                side_information = self.config[_experiment][p].get("side_information", {})
                side_information.update({k: self._set_path(self._base_folder_path_config,
                                                           v.format(self.config[_experiment][_dataset]))
                                         for k, v in side_information.items() if isinstance(v, str)})
                side_information = SimpleNamespace(**side_information)
                self.config[_experiment][p].update({k: self._set_path(self._base_folder_path_config,
                                                                      v.format(self.config[_experiment][_dataset]))
                                                    for k, v in self.config[_experiment][p].items() if
                                                    isinstance(v, str)})
                self.config[_experiment][p]["side_information"] = side_information
                self.config[_experiment][p][_dataloader] = self.config[_experiment][p].get(_dataloader, "DataSetLoader")
                setattr(self.base_namespace, p, SimpleNamespace(**self.config[_experiment][p]))
            elif p == _splitting and self.config[_experiment].get(p, {}):
                self.config[_experiment][p].update({k: self._set_path(self._base_folder_path_config,
                                                                      v.format(self.config[_experiment][_dataset]))
                                                    for k, v in self.config[_experiment][p].items() if
                                                    isinstance(v, str)})

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
                preprocessing_strategy = SimpleNamespace(**self.config[_experiment][p])
                self.config[_experiment][p] = preprocessing_strategy
                setattr(self.base_namespace, p, self.config[_experiment][p])
            elif p == _evaluation and self.config[_experiment].get(p, {}):
                complex_metrics = self.config[_experiment][p].get("complex_metrics", {})
                paired_ttest = self.config[_experiment][p].get("paired_ttest", {})
                wilcoxon_test = self.config[_experiment][p].get("wilcoxon_test", {})
                for complex_metric in complex_metrics:
                    complex_metric.update({k: self._set_path(self._base_folder_path_config,
                                                      v.format(self.config[_experiment][_dataset]))
                                    for k, v in complex_metric.items() if isinstance(v, str)})
                self.config[_experiment][p]["complex_metrics"] = complex_metrics
                self.config[_experiment][p]["paired_ttest"] = paired_ttest
                self.config[_experiment][p]["wilcoxon_test"] = wilcoxon_test
                setattr(self.base_namespace, p, SimpleNamespace(**self.config[_experiment][p]))
            elif p == _logger_config and not self.config[_experiment].get(p, False):
                setattr(self.base_namespace, p, f"{self._base_folder_path_elliot}/config/logger_config.yml")
            elif p == _log_folder and not self.config[_experiment].get(p, False):
                setattr(self.base_namespace, p, f"{self._base_folder_path_elliot}/../log/")
            elif p == _external_models_path and self.config[_experiment].get(p, False):
                self.config[_experiment][p] = self._set_path(self._base_folder_path_config, self.config[_experiment][p])
                setattr(self.base_namespace, p, self.config[_experiment][p])
            elif p == _config_test:
                setattr(self.base_namespace, p, self.config[_experiment].get(p, False))
            else:
                if self.config[_experiment].get(p):
                    setattr(self.base_namespace, p, self.config[_experiment][p])

    def fill_model(self):
        for key in self.config[_experiment][_models]:
            meta_model = self.config[_experiment][_models][key][_meta]
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
                            val_string = value[0].replace(" ","").split("(")[1].split(")")[0] if len(value[0].replace(" ","").split("(")) > 1 else None
                            val = [literal_eval(val_string) if val_string else None]
                            val.extend([literal_eval(val.replace(" ","").replace(")","")) if isinstance(val, str) else val for val in value[1:]])
                            val = [v for v in val if v is not None]
                            space_list.append((k, func_(k, *val)))
                        else:
                            # if not all([isinstance(v, str) for v in value]):
                            #     space_list.append((k, hp.choice(k, value)))
                            # else:
                            #     space_list.append((k, hp.choice(k, literal_eval(
                            #         "["+str(",".join([str(v) for v in value]))+"]")
                            #                                     )))
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
                yield key, model_name_space
