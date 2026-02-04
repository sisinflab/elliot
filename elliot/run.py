"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from typing import List, Optional
import copy
import os
import numpy as np

from elliot.dataset import DataSetLoader
from elliot.namespace import build_namespace, ExperimentConfig
from elliot.hyperoptimization import run_hyperopt
from elliot.result_handler.result_handler import ResultHandler, StatTest
from elliot.utils import logging as logging_project
from elliot.utils import set_device
from elliot.utils.folder import check_dir


print(u'''

  /\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\   /\\\\\\\\\\\\      /\\\\\\\\\\\\                         ''' + f'Version: {__version__}' + '''                              
  \\/\\\\\\///////////   \\////\\\\\\     \\////\\\\\\                                           
   \\/\\\\\\                 \\/\\\\\\        \\/\\\\\\      /\\\\\\                     /\\\\\\       
    \\/\\\\\\\\\\\\\\\\\\\\\\         \\/\\\\\\        \\/\\\\\\     \\///       /\\\\\\\\\\      /\\\\\\\\\\\\\\\\\\\\\\     
     \\/\\\\\\///////          \\/\\\\\\        \\/\\\\\\      /\\\\\\    /\\\\\\///\\\\\\   \\////\\\\\\////     
      \\/\\\\\\                 \\/\\\\\\        \\/\\\\\\     \\/\\\\\\   /\\\\\\  \\//\\\\\\     \\/\\\\\\    
       \\/\\\\\\                 \\/\\\\\\        \\/\\\\\\     \\/\\\\\\  \\//\\\\\\  /\\\\\\      \\/\\\\\\ /\\\\   
        \\/\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\   /\\\\\\\\\\\\\\\\\\   /\\\\\\\\\\\\\\\\\\  \\/\\\\\\   \\///\\\\\\\\\\/       \\//\\\\\\\\\\  
         \\///////////////   \\/////////   \\/////////   \\///      \\/////          \\/////    
         ''')


def run_experiment(config_path: str = "", config_overrides: Optional[List[str]] = None):
    config = build_namespace(config_path, config_overrides)

    if config.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

    if config.config_test:
        config_test(config)

    logging_project.init(config.path_logger_config, config.path_log_folder)
    logger = logging_project.get_logger("__main__")

    _configure_torch_device(config, logger)

    if config.version != __version__:
        logger.error(f'Your config file uses a different version of Elliot! '
                     f'In different versions of Elliot the results may slightly change due to progressive improvement! '
                     f'Some features could be deprecated! Download latest version at this link '
                     f'https://github.com/sisinflab/elliot/releases')
        raise Exception(
            'Version mismatch! In different versions of Elliot the results may slightly change due to progressive improvement!')

    logger.info("Start experiment")

    dataset_loader = DataSetLoader(config=config)
    data_test_list = dataset_loader.build()

    res_handler = ResultHandler(rel_threshold=config.evaluation.relevance_threshold)

    all_trials = {}

    for model_name, model_config in config.models.items():
        test_results = []
        test_trials = []
        all_trials[model_name] = []

        for test_fold_index, data_test in enumerate(data_test_list):
            logging_project.prepare_logger(model_name, config.path_log_folder)

            logger.info(f"Tuning begun for {model_name}\n")

            outcome = run_hyperopt(
                data_test=data_test,
                config=config,
                model_config=model_config,
                model_name=model_name,
                test_fold_index=test_fold_index
            )
            best_eval = outcome.best_eval

            ############################################
            best_model_loss = best_eval["loss"]
            best_model_params = best_eval["params"]
            best_model_results = best_eval["test_results"]
            ############################################

            # aggiunta a lista performance test
            test_results.append(best_eval)

            test_trials.append(outcome.all_trial_results)
            all_trials[model_name].append(outcome.all_trial_results)

            logger.info(f"Tuning ended for {model_name}")

            logger.info(f"Loss:\t{best_model_loss}")
            logger.info(f"Best Model params:\t{best_model_params}")
            logger.info(f"Best Model results:\t{best_model_results}")

        # Migliore sui test, aggiunta a performance totali
        min_val = np.argmin([i["loss"] for i in test_results])
        best_eval = test_results[min_val]

        results_config = config.results
        if results_config.save_fold_stats:
            _attach_fold_stats(best_eval, test_results, config)

        res_handler.add_oneshot_recommender(**best_eval)

        res_handler.add_trials(test_trials[min_val], name=model_name)
        all_trials[model_name] = all_trials[model_name][min_val]

    _save_outputs(res_handler, config)

    logger.info("End experiment")
    # TODO: check before to push only this feature!
    # logger.info("Start Post-Hoc scripts")

    # spec = importlib.util.spec_from_file_location("post_hoc", path.relpath(base.base_namespace.external_posthoc_path))
    # post_hoc = importlib.util.module_from_spec(spec)
    # sys.modules[spec.name] = post_hoc
    # spec.loader.exec_module(post_hoc)
    # post_hoc.run(data_test_list, all_trials)

    # logger.info("End Post-Hoc scripts")


def _attach_fold_stats(best_eval, fold_results, config):
    if len(fold_results) < 2:
        return
    sample = fold_results[0].get("test_results", {})
    if not sample:
        return
    metrics = list(next(iter(sample.values())).keys())

    def _aggregate(key, reducer):
        agg = {}
        for k in config.evaluation.cutoffs:
            agg[k] = {}
            for metric in metrics:
                values = [fold[key][k][metric] for fold in fold_results if k in fold[key]]
                if not values:
                    continue
                agg[k][metric] = float(reducer(values))
        return agg

    best_eval["test_mean_results"] = _aggregate("test_results", np.mean)
    best_eval["test_std_results"] = _aggregate("test_results", np.std)


def _save_outputs(res_handler, config):
    results_config = config.results
    output = config.path_output_rec_performance
    check_dir(output)

    if results_config.save_performance:
        res_handler.save_results(output=output, triplets=False)
    if results_config.save_performance_triplets:
        res_handler.save_results(output=output, triplets=True)

    if results_config.save_fold_stats:
        res_handler.save_results(output=output, key="test_mean_results", triplets=False)
        res_handler.save_results(output=output, key="test_std_results", triplets=False)
        if results_config.save_fold_stats_triplets:
            res_handler.save_results(output=output, key="test_mean_results", triplets=True)
            res_handler.save_results(output=output, key="test_std_results", triplets=True)

    if results_config.save_times:
        res_handler.save_times(output=output)

    if results_config.save_best_models:
        cutoffs = config.evaluation.cutoffs
        first_metric = (
            config.evaluation.simple_metrics[0]
            if config.evaluation.simple_metrics else ""
        )
        res_handler.save_best_models(output=output, default_metric=first_metric, default_k=cutoffs[0])

    if results_config.save_trials:
        res_handler.save_trials(output=output, formats=results_config.trials_formats)

    if results_config.save_statistical:
        if results_config.evaluation.paired_ttest:
            res_handler.save_statistical_results(StatTest.PairedTTest, output=output)
        if results_config.evaluation.wilcoxon_test:
            res_handler.save_statistical_results(StatTest.WilcoxonTest, output=output)


def config_test(config: ExperimentConfig):
    logging_project.init(config.path_logger_config, config.path_log_folder)
    logger = logging_project.get_logger("__main__")

    _configure_torch_device(config, logger)

    logger.info("Start config test")

    dataset_loader = DataSetLoader(config=config)
    data_test_list = dataset_loader.generate_dataobjects_mock()

    res_handler = ResultHandler(rel_threshold=config.evaluation.relevance_threshold)

    for model_name, model_config in config.models.items():
        test_results = []
        test_trials = []

        for test_fold_index, data_test in enumerate(data_test_list):
            model_config_mock = copy.deepcopy(model_config)
            model_config_mock.meta.verbose = False
            model_config_mock.meta.save_recs = False
            model_config_mock.meta.save_weights = False

            outcome = run_hyperopt(
                data_test=data_test,
                config=config,
                model_config=model_config_mock,
                model_name=model_name,
                test_fold_index=test_fold_index
            )

            test_results.append(outcome.best_eval)
            if outcome.trials is not None:
                test_trials.append(outcome.trials)

        min_val = np.argmin([i["loss"] for i in test_results])

        res_handler.add_oneshot_recommender(**test_results[min_val])
        res_handler.add_trials(test_trials[min_val])

    logger.info("End config test without issues")
    config.config_test = False


def _configure_torch_device(config, logger=None):
    requested = config.device or config.torch_device
    device = set_device(requested)
    if logger is not None:
        logger.info("Torch device selected", extra={"context": {"device": str(device)}})


if __name__ == '__main__':
    run_experiment("./config/config.yml")
