"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import importlib
import typing

from hyperopt import Trials, fmin

from namespace.namespace_model_builder import NameSpaceBuilder
from result_handler import ResultHandler
import hyperoptimization as ho
import numpy as np
from utils import logging as logging_project

_rstate = np.random.RandomState(42)


def run_experiment(config_path: str='./config/config.yml'):
    builder = NameSpaceBuilder(config_path)
    base = builder.base
    logging_project.init(base.base_namespace.path_logger_config, base.base_namespace.path_log_folder)
    logger = logging_project.get_logger("__main__")
    logger.warning("Test")
    res_handler = ResultHandler()
    dataloader_class = getattr(importlib.import_module("dataset"), base.base_namespace.data_config.dataloader)
    dataloader = dataloader_class(config=base.base_namespace)
    data_test_list = dataloader.generate_dataobjects()
    for key, model_base in builder.models():
        test_results = []
        test_trials = []
        for data_test in data_test_list:
            logging_project.prepare_logger(key, base.base_namespace.path_log_folder)
            model_class = getattr(importlib.import_module("recommender"), key)
            print("\n********************************")
            print(f"Tuning begun for {model_class.__name__}\n")
            model_placeholder = ho.ModelCoordinator(data_test, base.base_namespace, model_base, model_class)
            if isinstance(model_base, tuple):
                # model_placeholder = ho.ModelCoordinator(dataloader, base.base_namespace, model_base[0], model_class)
                trials = Trials()
                best = fmin(model_placeholder.objective,
                            space=model_base[1],
                            algo=model_base[3],
                            trials=trials,
                            rstate=_rstate,
                            max_evals=model_base[2])
                # res_handler.add_multishot_recommender(trials)
                min_val = np.argmin([i["result"]["loss"] for i in trials._trials])
                best_model_loss = trials._trials[min_val]["result"]["loss"]
                best_model_params = trials._trials[min_val]["result"]["params"]
                best_model_results = trials._trials[min_val]["result"]["test_results"]

                # aggiunta a lista performance test
                test_results.append(trials._trials[min_val]["result"])
                test_trials.append(trials)
            else:
                single = model_placeholder.single()
                # res_handler.add_oneshot_recommender(**single)
                best_model_loss = single["loss"]
                best_model_params = single["params"]
                best_model_results = single["test_results"]

                # aggiunta a lista performance test
                test_results.append(single)

            print(f"Loss: {best_model_loss}")
            print(f"Best Model params: {best_model_params}")
            print(f"Best Model results: {best_model_results}")
            print(f"\nTuning ended for {model_class.__name__}")
            print("********************************\n")

        # Media sui test, aggiunta a performance totali
        min_val = np.argmin([i["loss"] for i in test_results])
        res_handler.add_oneshot_recommender(**test_results[min_val])
        if isinstance(model_base, tuple):
            res_handler.add_trials(test_trials[min_val])

    # res_handler.save_results(output=base.base_namespace.path_output_rec_performance)
    res_handler.save_trials(output=base.base_namespace.path_output_rec_performance)
    res_handler.save_best_results(output=base.base_namespace.path_output_rec_performance)
    if base.base_namespace.evaluation.paired_ttest:
        res_handler.save_best_statistical_results(output=base.base_namespace.path_output_rec_performance)


if __name__ == '__main__':
    builder = NameSpaceBuilder('./config/config.yml')
    base = builder.base
    logging_project.init(base.base_namespace.path_logger_config, base.base_namespace.path_log_folder)
    logger = logging_project.get_logger(__name__)
    logger.warning("Test")
    res_handler = ResultHandler()
    dataloader_class = getattr(importlib.import_module("dataset"), base.base_namespace.data_config.dataloader)
    dataloader = dataloader_class(config=base.base_namespace)
    data_test_list = dataloader.generate_dataobjects()
    for key, model_base in builder.models():
        test_results = []
        test_trials = []
        for data_test in data_test_list:
            logging_project.prepare_logger(key, base.base_namespace.path_log_folder)
            model_class = getattr(importlib.import_module("recommender"), key)
            print("\n********************************")
            print(f"Tuning begun for {model_class.__name__}\n")
            model_placeholder = ho.ModelCoordinator(data_test, base.base_namespace, model_base, model_class)
            if isinstance(model_base, tuple):
                # model_placeholder = ho.ModelCoordinator(dataloader, base.base_namespace, model_base[0], model_class)
                trials = Trials()
                best = fmin(model_placeholder.objective,
                            space=model_base[1],
                            algo=model_base[3],
                            trials=trials,
                            rstate=_rstate,
                            max_evals=model_base[2])
                # res_handler.add_multishot_recommender(trials)
                min_val = np.argmin([i["result"]["loss"] for i in trials._trials])
                best_model_loss = trials._trials[min_val]["result"]["loss"]
                best_model_params = trials._trials[min_val]["result"]["params"]
                best_model_results = trials._trials[min_val]["result"]["test_results"]

                # aggiunta a lista performance test
                test_results.append(trials._trials[min_val]["result"])
                test_trials.append(trials)
            else:
                single = model_placeholder.single()
                # res_handler.add_oneshot_recommender(**single)
                best_model_loss = single["loss"]
                best_model_params = single["params"]
                best_model_results = single["test_results"]

                # aggiunta a lista performance test
                test_results.append(single)

            print(f"Loss: {best_model_loss}")
            print(f"Best Model params: {best_model_params}")
            print(f"Best Model results: {best_model_results}")
            print(f"\nTuning ended for {model_class.__name__}")
            print("********************************\n")

        # Media sui test, aggiunta a performance totali
        min_val = np.argmin([i["loss"] for i in test_results])
        res_handler.add_oneshot_recommender(**test_results[min_val])
        if isinstance(model_base, tuple):
            res_handler.add_trials(test_trials[min_val])

    # res_handler.save_results(output=base.base_namespace.path_output_rec_performance)
    res_handler.save_trials(output=base.base_namespace.path_output_rec_performance)
    res_handler.save_best_results(output=base.base_namespace.path_output_rec_performance)
    if base.base_namespace.evaluation.paired_ttest:
        res_handler.save_best_statistical_results(output=base.base_namespace.path_output_rec_performance)

