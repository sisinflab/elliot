"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import os
from types import SimpleNamespace

from hyperopt import hp
from yaml import FullLoader as FullLoader
from yaml import load
from collections import OrderedDict
from utils.folder import manage_directories
import hyperoptimization as ho

_experiment = 'experiment'

_data_config = "data_config"
_splitting = "splitting"
_dataset = 'dataset'
_dataloader = 'dataloader'
_weights = 'path_output_rec_weight'
_performance = 'path_output_rec_performance'
_logger_config = 'path_logger_config'
_log_folder = 'path_log_folder'
_verbose = 'verbose'
_recs = 'path_output_rec_result'
_top_k = 'top_k'
_metrics = 'metrics'
_relevance = 'relevance'
_paired_ttest = 'paired_ttest'
_models = 'models'
_recommender = 'recommender'
_gpu = 'gpu'
_hyper_max_evals = 'hyper_max_evals'
_hyper_opt_alg = 'hyper_opt_alg'
_data_paths = 'data_paths'
_meta = 'meta'


class NameSpaceModel:
    def __init__(self, config_path):
        self.base_namespace = SimpleNamespace()

        self.config_file = open(config_path)
        self.config = load(self.config_file, Loader=FullLoader)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config[_experiment][_gpu])

    def fill_base(self):

        # for path in self.config[_experiment][_data_paths].keys():
        #     self.config[_experiment][_data_paths][path] = \
        #         self.config[_experiment][_data_paths][path].format(self.config[_experiment][_dataset])

        self.config[_experiment][_recs] = self.config[_experiment][_recs] \
            .format(self.config[_experiment][_dataset])
        self.config[_experiment][_weights] = self.config[_experiment][_weights] \
            .format(self.config[_experiment][_dataset])
        self.config[_experiment][_performance] = self.config[_experiment][_performance] \
            .format(self.config[_experiment][_dataset])

        self.config[_experiment][_paired_ttest] = self.config[_experiment].get(_paired_ttest, False)
        self.config[_experiment][_dataloader] = self.config[_experiment].get(_dataloader, "DataSetLoader")

        manage_directories(self.config[_experiment][_recs], self.config[_experiment][_weights],
                           self.config[_experiment][_performance])

        for p in [_data_config, _weights, _recs, _dataset, _top_k, _metrics, _relevance, _paired_ttest, _performance, _logger_config,
                  _log_folder, _dataloader, _splitting]:
            if p == _data_config:
                side_information = self.config[_experiment][p].get("side_information", {})
                side_information.update({k: v.format(self.config[_experiment][_dataset])
                                               for k, v in side_information.items() if isinstance(v, str)})
                side_information = SimpleNamespace(**side_information)
                self.config[_experiment][p].update({k: v.format(self.config[_experiment][_dataset])
                                               for k, v in self.config[_experiment][p].items() if isinstance(v, str)})
                self.config[_experiment][p]["side_information"] = side_information
                self.config[_experiment][p][_dataloader] = self.config[_experiment][p].get(_dataloader, "DataSetLoader")
                setattr(self.base_namespace, p, SimpleNamespace(**self.config[_experiment][p]))
            elif p == _splitting:
                if self.config[_experiment].get(p, {}):
                    self.config[_experiment][p].update({k: v.format(self.config[_experiment][_dataset])
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

            else:
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
                        space_list.append((k, hp.choice(k, value)))
                _SPACE = OrderedDict(space_list)
                _max_evals = meta_model[_hyper_max_evals]
                _opt_alg = ho.parse_algorithms(meta_model[_hyper_opt_alg])
                yield key, (model_name_space, _SPACE, _max_evals, _opt_alg)
            else:
                yield key, model_name_space
