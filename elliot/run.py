"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import importlib

from hyperopt import Trials, fmin

from namespace.namespace_model_builder import NameSpaceBuilder
from result_handler import ResultHandler
import hyperoptimization as ho
import numpy as np
from utils import logging as logging_project

_rstate = np.random.RandomState(42)

if __name__ == '__main__':
    builder = NameSpaceBuilder('./config/config.yml')
    base = builder.base
    logging_project.init(base.base_namespace.path_logger_config, base.base_namespace.path_log_folder)
    logger = logging_project.getLogger(__name__)
    logger.warning("Test")
    res_handler = ResultHandler()
    for key, model_base in builder.models():
        model_class = getattr(importlib.import_module("recommender"), key)
        if isinstance(model_base, tuple):
            model_placeholder = ho.ModelCoordinator(base.base_namespace, model_base[0], model_class)
            trials = Trials()
            best = fmin(model_placeholder.objective,
                        space=model_base[1],
                        algo=model_base[3],
                        trials=trials,
                        rstate=_rstate,
                        max_evals=model_base[2])
            res_handler.add_multishot_recommender(trials)
            min_val = np.argmin([i["result"]["loss"] for i in trials._trials])
            best_model_loss = trials._trials[min_val]["result"]["loss"]
            best_model_params = trials._trials[min_val]["result"]["params"]
            best_model_results = trials._trials[min_val]["result"]["results"]
        else:
            model = model_class(config=base.base_namespace, params=model_base)
            model.train()
            res_handler.add_oneshot_recommender(model.name, model.get_loss(), model.get_params(), model.get_results())
            best_model_loss = model.get_loss()
            best_model_params = model.get_params()
            best_model_results = model.get_results()
        print(f"Loss: {best_model_loss}")
        print(f"Best Model params: {best_model_params}")
        print(f"Best Model results: {best_model_results}")
        print(f"\nHyperparameter tuning ended for {model_class.__name__}")
        print("********************************\n")
    res_handler.save_results(output=base.base_namespace.path_output_rec_performance)
    res_handler.save_best_results(output=base.base_namespace.path_output_rec_performance)

