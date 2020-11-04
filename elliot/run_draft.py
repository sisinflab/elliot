import importlib
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
import typing as t

from yaml import FullLoader as FullLoader
from yaml import load

from hyperopt import hp, tpe, fmin, Trials

from hyperoptimization import ModelCoordinator

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
_rstate = np.random.RandomState(42)

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
        # model_namespace = SimpleNamespace(
        #     lr=0.01,
        #     epochs=1,
        #     embed_k=20,
        #     batch_size=512,
        #     bias_regularization=1,
        #     user_regularization=1,
        #     positive_item_regularization=1,
        #     negative_item_regularization=1,
        #     update_negative_item_factors=True,
        #     update_users=True,
        #     update_items=True,
        #     update_bias=True
        # )
        _SPACE = OrderedDict([('lr', hp.loguniform('lr', np.log(0.01), np.log(0.5))),
                             # ('max_depth', hp.choice('max_depth', range(1, 30, 1))),
                             ('embed_k', hp.choice('embed_k', range(20, 50, 10))),
                             # ('min_data_in_leaf', hp.choice('min_data_in_leaf', range(10, 1000, 1))),
                             # ('feature_fraction', hp.uniform('feature_fraction', 0.1, 1.0)),
                             # ('subsample', hp.uniform('subsample', 0.1, 1.0))
                             ])
        _max_evals = 2
        # import hyperopt.pyll.stochastic
        # print(hyperopt.pyll.stochastic.sample(SPACE))

        model_class: t.ClassVar = getattr(importlib.import_module(_recommender), key)

        model_placeholder = ModelCoordinator(base, SimpleNamespace(**config[_experiment][_models][key]), model_class)
        trials = Trials()
        best = fmin(model_placeholder.objective,
                    space=_SPACE,
                    algo=tpe.suggest,
                    trials=trials,
                    rstate=_rstate,
                    max_evals=_max_evals)

        print(best)
        min_val = np.argmin([i["result"]["loss"] for i in trials._trials])
        best_model_loss = trials._trials[min_val]["result"]["loss"]
        best_model_params = trials._trials[min_val]["result"]["params"]
        best_model_results = trials._trials[min_val]["result"]["results"]
        print(f"Loss: {best_model_loss}")
        print(f"Best Model params: {best_model_params}")
        print(f"Best Model results: {best_model_results}")
        print(f"\nHyperparameter tuning ended for {model_class.__name__}")
        print("********************************\n")





