import os
from types import SimpleNamespace

from hyperopt import hp
from yaml import FullLoader as FullLoader
from yaml import load
from collections import OrderedDict
from utils.folder import manage_directories
import hyperoptimization as ho

_experiment = 'experiment'
_training_set = 'path_train_data'
_validation_set = 'path_validation_data'
_dataset = 'dataset'
_test_set = 'path_test_data'
_weights = 'path_output_rec_weight'
_performance = 'path_output_rec_performance'
_recs = 'path_output_rec_result'
_features = 'path_feature_data'
_top_k = 'top_k'
_metrics = 'metrics'
_relevance = 'relevance'
_models = 'models'
_recommender = 'recommender'
_gpu = 'gpu'
_hyper_max_evals = 'hyper_max_evals'
_hyper_opt_alg = 'hyper_opt_alg'


class NameSpaceModel:
    def __init__(self, config_path):
        self.base_namespace = SimpleNamespace()

        self.config_file = open(config_path)
        self.config = load(self.config_file, Loader=FullLoader)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config[_experiment][_gpu])

    def fill_base(self):

        self.config[_experiment][_training_set] = self.config[_experiment][_training_set] \
            .format(self.config[_experiment][_dataset])
        self.config[_experiment][_validation_set] = self.config[_experiment][_validation_set] \
            .format(self.config[_experiment][_dataset])
        self.config[_experiment][_test_set] = self.config[_experiment][_test_set] \
            .format(self.config[_experiment][_dataset])
        self.config[_experiment][_features] = self.config[_experiment][_features]\
            .format(self.config[_experiment][_dataset])

        self.config[_experiment][_recs] = self.config[_experiment][_recs] \
            .format(self.config[_experiment][_dataset])
        self.config[_experiment][_weights] = self.config[_experiment][_weights] \
            .format(self.config[_experiment][_dataset])
        self.config[_experiment][_performance] = self.config[_experiment][_performance] \
            .format(self.config[_experiment][_dataset])

        manage_directories(self.config[_experiment][_recs], self.config[_experiment][_weights],
                           self.config[_experiment][_performance])

        for p in [_training_set, _validation_set, _test_set, _weights, _features, _recs, _dataset, _top_k, _metrics,
                  _relevance, _performance]:
            setattr(self.base_namespace, p, self.config[_experiment][p])

    def fill_model(self):
        for key in self.config[_experiment][_models]:
            if any(isinstance(value, list) for value in self.config[_experiment][_models][key].values()):
                space_list = []
                for k, value in self.config[_experiment][_models][key].items():
                    if isinstance(value, list):
                        space_list.append((k, hp.choice(k, value)))
                _SPACE = OrderedDict(space_list)
                _max_evals = self.config[_experiment][_models][key][_hyper_max_evals]
                _opt_alg = ho.parse_algorithms(self.config[_experiment][_models][key][_hyper_opt_alg])
                yield key, (SimpleNamespace(**self.config[_experiment][_models][key]), _SPACE, _max_evals, _opt_alg)
            else:
                yield key, SimpleNamespace(**self.config[_experiment][_models][key])
