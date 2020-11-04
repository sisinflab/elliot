import importlib
from types import SimpleNamespace

from yaml import FullLoader as FullLoader
from yaml import load

from utils.folder import manage_directories

_experiment = 'experiment'
_training_set = 'training_set'
_validation_set = 'validation_set'
_dataset = 'dataset'
_test_set = 'test_set'
_weights = 'weights'
_recs = 'recs'
_features = 'features'
_top_k = 'top_k'
_metrics = 'metrics'
_relevance = 'relevance'
_models = 'models'
_recommender = 'recommender'

if __name__ == '__main__':
    config_file = open('./config/config.yml')
    config = load(config_file, Loader=FullLoader)

    config[_experiment][_training_set] = config[_experiment][_training_set]\
        .format(config[_experiment][_dataset])
    config[_experiment][_validation_set] = config[_experiment][_validation_set] \
        .format(config[_experiment][_dataset])
    config[_experiment][_test_set] = config[_experiment][_test_set] \
        .format(config[_experiment][_dataset])
    config[_experiment][_features] = config[_experiment][_features] \
        .format(config[_experiment][_dataset])

    config[_experiment][_recs] = config[_experiment][_recs] \
        .format(config[_experiment][_dataset])
    config[_experiment][_weights] = config[_experiment][_weights] \
        .format(config[_experiment][_dataset])

    manage_directories(config[_experiment][_recs], config[_experiment][_weights])

    base = SimpleNamespace(
        path_train_data=config[_experiment][_training_set],
        path_validation_data=config[_experiment][_validation_set],
        path_test_data=config[_experiment][_test_set],
        path_feature_data=config[_experiment][_features],
        path_output_rec_result=config[_experiment][_recs],
        path_output_rec_weight=config[_experiment][_weights],
        dataset=config[_experiment][_dataset],
        top_k=config[_experiment][_top_k],
        metrics=config[_experiment][_metrics],
        relevance=config[_experiment][_relevance],
    )

    for key in config[_experiment][_models]:
        model_class = getattr(importlib.import_module(_recommender), key)
        model = model_class(config=base, params=SimpleNamespace(**config[_experiment][_models][key]))
        model.train()