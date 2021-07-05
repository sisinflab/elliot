"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import importlib
import sys
from os import path

import numpy as np
from hyperopt import Trials, fmin

import elliot.hyperoptimization as ho
from elliot.namespace.namespace_model_builder import NameSpaceBuilder
from elliot.result_handler.result_handler import ResultHandler, HyperParameterStudy, StatTest
from elliot.utils import logging as logging_project

_rstate = np.random.RandomState(42)
here = path.abspath(path.dirname(__file__))

print(u'''
__/\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\___/\\\\\\\\\\\\______/\\\\\\\\\\\\_________________________________________        
 _\\/\\\\\\///////////___\\////\\\\\\_____\\////\\\\\\_________________________________________       
  _\\/\\\\\\_________________\\/\\\\\\________\\/\\\\\\______/\\\\\\_____________________/\\\\\\______      
   _\\/\\\\\\\\\\\\\\\\\\\\\\_________\\/\\\\\\________\\/\\\\\\_____\\///_______/\\\\\\\\\\______/\\\\\\\\\\\\\\\\\\\\\\_     
    _\\/\\\\\\///////__________\\/\\\\\\________\\/\\\\\\______/\\\\\\____/\\\\\\///\\\\\\___\\////\\\\\\////__    
     _\\/\\\\\\_________________\\/\\\\\\________\\/\\\\\\_____\\/\\\\\\___/\\\\\\__\\//\\\\\\_____\\/\\\\\\______   
      _\\/\\\\\\_________________\\/\\\\\\________\\/\\\\\\_____\\/\\\\\\__\\//\\\\\\__/\\\\\\______\\/\\\\\\_/\\\\__  
       _\\/\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\___/\\\\\\\\\\\\\\\\\\___/\\\\\\\\\\\\\\\\\\__\\/\\\\\\___\\///\\\\\\\\\\/_______\\//\\\\\\\\\\___ 
        _\\///////////////___\\/////////___\\/////////___\\///______\\/////__________\\/////____''')

print(f'Version Number: {__version__}')


def run_experiment(config_path: str = ''):
    builder = NameSpaceBuilder(config_path, here, path.abspath(path.dirname(config_path)))
    base = builder.base
    config_test(builder, base)
    logging_project.init(base.base_namespace.path_logger_config, base.base_namespace.path_log_folder)
    logger = logging_project.get_logger("__main__")

    if base.base_namespace.version != __version__:
        logger.error(f'Your config file use a different version of Elliot! '
                     f'In different versions of Elliot the results may slightly change due to progressive improvement! '
                     f'Some feature could be deprecated! Download latest version at this link '
                     f'https://github.com/sisinflab/elliot/releases')
        raise Exception(
            'Version mismatch! In different versions of Elliot the results may slightly change due to progressive improvement!')

    logger.info("Start experiment")
    base.base_namespace.evaluation.relevance_threshold = getattr(base.base_namespace.evaluation, "relevance_threshold",
                                                                 0)
    res_handler = ResultHandler(rel_threshold=base.base_namespace.evaluation.relevance_threshold)
    hyper_handler = HyperParameterStudy(rel_threshold=base.base_namespace.evaluation.relevance_threshold)
    dataloader_class = getattr(importlib.import_module("elliot.dataset"), base.base_namespace.data_config.dataloader)
    dataloader = dataloader_class(config=base.base_namespace)
    data_test_list = dataloader.generate_dataobjects()
    for key, model_base in builder.models():
        test_results = []
        test_trials = []
        for test_fold_index, data_test in enumerate(data_test_list):
            logging_project.prepare_logger(key, base.base_namespace.path_log_folder)
            if key.startswith("external."):
                spec = importlib.util.spec_from_file_location("external",
                                                              path.relpath(base.base_namespace.external_models_path))
                external = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = external
                spec.loader.exec_module(external)
                model_class = getattr(importlib.import_module("external"), key.split(".", 1)[1])
            else:
                model_class = getattr(importlib.import_module("elliot.recommender"), key)

            model_placeholder = ho.ModelCoordinator(data_test, base.base_namespace, model_base, model_class,
                                                    test_fold_index)
            if isinstance(model_base, tuple):
                logger.info(f"Tuning begun for {model_class.__name__}\\n")
                trials = Trials()
                fmin(model_placeholder.objective,
                     space=model_base[1],
                     algo=model_base[3],
                     trials=trials,
                     verbose=False,
                     rstate=_rstate,
                     max_evals=model_base[2])

                # argmin relativo alla combinazione migliore di iperparametri
                min_val = np.argmin([i["result"]["loss"] for i in trials._trials])
                ############################################
                best_model_loss = trials._trials[min_val]["result"]["loss"]
                best_model_params = trials._trials[min_val]["result"]["params"]
                best_model_results = trials._trials[min_val]["result"]["test_results"]
                ############################################

                # aggiunta a lista performance test
                test_results.append(trials._trials[min_val]["result"])
                test_trials.append(trials)
                logger.info(f"Tuning ended for {model_class.__name__}")
            else:
                logger.info(f"Training begun for {model_class.__name__}\\n")
                single = model_placeholder.single()

                ############################################
                best_model_loss = single["loss"]
                best_model_params = single["params"]
                best_model_results = single["test_results"]
                ############################################

                # aggiunta a lista performance test
                test_results.append(single)
                logger.info(f"Training ended for {model_class.__name__}")

            logger.info(f"Loss:\\t{best_model_loss}")
            logger.info(f"Best Model params:\\t{best_model_params}")
            logger.info(f"Best Model results:\\t{best_model_results}")

        # Migliore sui test, aggiunta a performance totali
        min_val = np.argmin([i["loss"] for i in test_results])

        res_handler.add_oneshot_recommender(**test_results[min_val])

        if isinstance(model_base, tuple):
            hyper_handler.add_trials(test_trials[min_val])

    # res_handler.save_results(output=base.base_namespace.path_output_rec_performance)
    hyper_handler.save_trials(output=base.base_namespace.path_output_rec_performance)
    res_handler.save_best_results(output=base.base_namespace.path_output_rec_performance)
    cutoff_k = getattr(base.base_namespace.evaluation, "cutoffs", [base.base_namespace.top_k])
    cutoff_k = cutoff_k if isinstance(cutoff_k, list) else [cutoff_k]
    first_metric = base.base_namespace.evaluation.simple_metrics[
        0] if base.base_namespace.evaluation.simple_metrics else ""
    res_handler.save_best_models(output=base.base_namespace.path_output_rec_performance, default_metric=first_metric,
                                 default_k=cutoff_k)
    if hasattr(base.base_namespace,
               "print_results_as_triplets") and base.base_namespace.print_results_as_triplets == True:
        res_handler.save_best_results_as_triplets(output=base.base_namespace.path_output_rec_performance)
        hyper_handler.save_trials_as_triplets(output=base.base_namespace.path_output_rec_performance)
    if hasattr(base.base_namespace.evaluation, "paired_ttest") and base.base_namespace.evaluation.paired_ttest:
        res_handler.save_best_statistical_results(stat_test=StatTest.PairedTTest,
                                                  output=base.base_namespace.path_output_rec_performance)
    if hasattr(base.base_namespace.evaluation, "wilcoxon_test") and base.base_namespace.evaluation.wilcoxon_test:
        res_handler.save_best_statistical_results(stat_test=StatTest.WilcoxonTest,
                                                  output=base.base_namespace.path_output_rec_performance)

    logger.info("End experiment")


def _reset_verbose_option(model):
    if isinstance(model, tuple):
        model[0].meta.verbose = False
        model[0].meta.save_recs = False
        model[0].meta.save_weights = False
    else:
        model.meta.verbose = False
        model.meta.save_recs = False
        model.meta.save_weights = False
    return model


def config_test(builder, base):
    if base.base_namespace.config_test:
        logging_project.init(base.base_namespace.path_logger_config, base.base_namespace.path_log_folder)
        logger = logging_project.get_logger("__main__")
        logger.info("Start config test")
        base.base_namespace.evaluation.relevance_threshold = getattr(base.base_namespace.evaluation,
                                                                     "relevance_threshold", 0)
        res_handler = ResultHandler(rel_threshold=base.base_namespace.evaluation.relevance_threshold)
        hyper_handler = HyperParameterStudy(rel_threshold=base.base_namespace.evaluation.relevance_threshold)
        dataloader_class = getattr(importlib.import_module("elliot.dataset"),
                                   base.base_namespace.data_config.dataloader)
        dataloader = dataloader_class(config=base.base_namespace)
        data_test_list = dataloader.generate_dataobjects_mock()
        for key, model_base in builder.models():
            test_results = []
            test_trials = []
            for data_test in data_test_list:
                if key.startswith("external."):
                    spec = importlib.util.spec_from_file_location("external",
                                                                  path.relpath(
                                                                      base.base_namespace.external_models_path))
                    external = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = external
                    spec.loader.exec_module(external)
                    model_class = getattr(importlib.import_module("external"), key.split(".", 1)[1])
                else:
                    model_class = getattr(importlib.import_module("elliot.recommender"), key)

                model_base_mock = model_base
                model_base_mock = _reset_verbose_option(model_base_mock)
                model_placeholder = ho.ModelCoordinator(data_test, base.base_namespace, model_base_mock, model_class)
                if isinstance(model_base, tuple):
                    trials = Trials()
                    fmin(model_placeholder.objective,
                         space=model_base_mock[1],
                         algo=model_base_mock[3],
                         trials=trials,
                         rstate=_rstate,
                         max_evals=model_base_mock[2])

                    min_val = np.argmin([i["result"]["loss"] for i in trials._trials])

                    test_results.append(trials._trials[min_val]["result"])
                    test_trials.append(trials)
                else:
                    single = model_placeholder.single()

                    test_results.append(single)

            min_val = np.argmin([i["loss"] for i in test_results])

            res_handler.add_oneshot_recommender(**test_results[min_val])

            if isinstance(model_base, tuple):
                hyper_handler.add_trials(test_trials[min_val])
        logger.info("End config test without issues")
    base.base_namespace.config_test = False


if __name__ == '__main__':
    run_experiment("./config/config.yml")
