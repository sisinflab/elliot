"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import importlib
from types import SimpleNamespace
from typing import List, Optional

import numpy as np

import elliot.hyperoptimization as ho
from elliot.namespace.namespace_model_builder import NameSpaceBuilder
from elliot.result_handler.result_handler import ResultHandler, StatTest
from elliot.recommender.utils import set_device, get_device
from elliot.utils import logging as logging_project
from elliot.utils.folder import parent_dir, check_dir
from elliot.utils.model_resolver import resolve_model_class

here = parent_dir(__file__)

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

def run_experiment(config_path: str = '', config_overrides: Optional[List[str]] = None):
    builder = NameSpaceBuilder(
        config_path, here, parent_dir(config_path), config_overrides=config_overrides
    )
    base = builder.base
    config_test(builder, base)
    logging_project.init(base.base_namespace.path_logger_config, base.base_namespace.path_log_folder)
    logger = logging_project.get_logger("__main__")
    _configure_torch_device(base.base_namespace, logger)

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
    rstate = np.random.default_rng(seed=getattr(base.base_namespace, "random_seed", 42))
    engine = ho.HyperOptEngine(rstate=rstate)
    runner = ho.HyperoptRunner(engine)
    dataloader_class = getattr(importlib.import_module("elliot.dataset"), 'DataSetLoader')
    dataloader = dataloader_class(config=base.base_namespace)
    data_test_list = dataloader.build()
    all_trials = {}
    for key, model_base in builder.models():
        test_results = []
        test_trials = []
        all_trials[key] = []
        for test_fold_index, data_test in enumerate(data_test_list):
            logging_project.prepare_logger(key, base.base_namespace.path_log_folder)
            model_class = resolve_model_class(key, base.base_namespace, here)

            if isinstance(model_base, tuple):
                logger.info(f"Tuning begun for {model_class.__name__}\n")
            else:
                logger.info(f"Training begun for {model_class.__name__}\n")

            outcome = runner.run(data_test, base.base_namespace, model_base, model_class, test_fold_index)
            best_eval = outcome.best_eval

            ############################################
            best_model_loss = best_eval["loss"]
            best_model_params = best_eval["params"]
            best_model_results = best_eval["test_results"]
            ############################################

            # aggiunta a lista performance test
            test_results.append(best_eval)
            if outcome.trials is not None:
                test_trials.append(outcome.all_trial_results)
                all_trials[key].append(outcome.all_trial_results)
                logger.info(f"Tuning ended for {model_class.__name__}")
            else:
                test_trials.append(outcome.all_trial_results)
                all_trials[key].append(outcome.all_trial_results)
                logger.info(f"Training ended for {model_class.__name__}")

            logger.info(f"Loss:\t{best_model_loss}")
            logger.info(f"Best Model params:\t{best_model_params}")
            logger.info(f"Best Model results:\t{best_model_results}")

        min_val = np.argmin([i["loss"] for i in test_results])
        best_eval = test_results[min_val]

        results_cfg = _results_config(base.base_namespace)
        if results_cfg.save_fold_stats:
            _attach_fold_stats(best_eval, test_results, base)

        res_handler.add_oneshot_recommender(**best_eval)

        if isinstance(model_base, tuple):
            res_handler.add_trials(test_trials[min_val], name=key)
        all_trials[key] = all_trials[key][min_val]

    _save_outputs(res_handler, base.base_namespace)

    logger.info("End experiment")
    # TODO: check before to push only this feature!
    # logger.info("Start Post-Hoc scripts")

    # spec = importlib.util.spec_from_file_location("post_hoc", path.relpath(base.base_namespace.external_posthoc_path))
    # post_hoc = importlib.util.module_from_spec(spec)
    # sys.modules[spec.name] = post_hoc
    # spec.loader.exec_module(post_hoc)
    # post_hoc.run(data_test_list, all_trials)

    # logger.info("End Post-Hoc scripts")


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


def _results_config(base_namespace):
    cfg = getattr(base_namespace, "results", None)
    if isinstance(cfg, dict):
        cfg = SimpleNamespace(**cfg)
    if cfg is None:
        cfg = SimpleNamespace()
    return SimpleNamespace(
        save_performance=getattr(cfg, "save_performance", True),
        save_performance_triplets=getattr(cfg, "save_performance_triplets", getattr(base_namespace, "print_results_as_triplets", False)),
        save_times=getattr(cfg, "save_times", True),
        save_best_models=getattr(cfg, "save_best_models", True),
        save_trials=getattr(cfg, "save_trials", True),
        trials_formats=getattr(cfg, "trials_formats", ["json", "tsv"]),
        save_fold_stats=getattr(cfg, "save_fold_stats", True),
        save_fold_stats_triplets=getattr(cfg, "save_fold_stats_triplets", False),
        save_statistical=getattr(cfg, "save_statistical", False),
    )


def _attach_fold_stats(best_eval, fold_results, base_namespace):
    if len(fold_results) < 2:
        return
    cutoffs = getattr(base_namespace.evaluation, "cutoffs", [base_namespace.top_k])
    cutoffs = cutoffs if isinstance(cutoffs, list) else [cutoffs]
    sample = fold_results[0].get("test_results", {})
    if not sample:
        return
    metrics = list(next(iter(sample.values())).keys())

    def _aggregate(key, reducer):
        agg = {}
        for k in cutoffs:
            agg[k] = {}
            for metric in metrics:
                values = [fold[key][k][metric] for fold in fold_results if k in fold[key]]
                if not values:
                    continue
                agg[k][metric] = float(reducer(values))
        return agg

    best_eval["test_mean_results"] = _aggregate("test_results", np.mean)
    best_eval["test_std_results"] = _aggregate("test_results", np.std)


def _save_outputs(res_handler, base_namespace):
    cfg = _results_config(base_namespace)
    output = base_namespace.path_output_rec_performance
    check_dir(output)

    if cfg.save_performance:
        res_handler.save_results(output=output, triplets=False)
    if cfg.save_performance_triplets:
        res_handler.save_results(output=output, triplets=True)

    if cfg.save_fold_stats:
        res_handler.save_results(output=output, key="test_mean_results", triplets=False)
        res_handler.save_results(output=output, key="test_std_results", triplets=False)
        if cfg.save_fold_stats_triplets:
            res_handler.save_results(output=output, key="test_mean_results", triplets=True)
            res_handler.save_results(output=output, key="test_std_results", triplets=True)

    if cfg.save_times:
        res_handler.save_times(output=output)

    if cfg.save_best_models:
        cutoffs = getattr(base_namespace.evaluation, "cutoffs", [base_namespace.top_k])
        cutoffs = cutoffs if isinstance(cutoffs, list) else [cutoffs]
        first_metric = base_namespace.evaluation.simple_metrics[0] if base_namespace.evaluation.simple_metrics else ""
        res_handler.save_best_models(output=output, default_metric=first_metric, default_k=cutoffs[0])

    if cfg.save_trials:
        res_handler.save_trials(output=output, formats=cfg.trials_formats)

    if cfg.save_statistical:
        if getattr(base_namespace.evaluation, "paired_ttest", False):
            res_handler.save_statistical_results(StatTest.PairedTTest, output=output)
        if getattr(base_namespace.evaluation, "wilcoxon_test", False):
            res_handler.save_statistical_results(StatTest.WilcoxonTest, output=output)


def config_test(builder, base):
    if base.base_namespace.config_test:
        logging_project.init(base.base_namespace.path_logger_config, base.base_namespace.path_log_folder)
        logger = logging_project.get_logger("__main__")
        _configure_torch_device(base.base_namespace, logger)
        logger.info("Start config test")
        base.base_namespace.evaluation.relevance_threshold = getattr(base.base_namespace.evaluation,
                                                                     "relevance_threshold", 0)
        res_handler = ResultHandler(rel_threshold=base.base_namespace.evaluation.relevance_threshold)
        rstate = np.random.default_rng(seed=getattr(base.base_namespace, "random_seed", 42))
        engine = ho.HyperOptEngine(rstate=rstate)
        runner = ho.HyperoptRunner(engine)
        dataloader_class = getattr(importlib.import_module("elliot.dataset"),
                                   base.base_namespace.data_config.dataloader)
        dataloader = dataloader_class(config=base.base_namespace)
        data_test_list = dataloader.generate_dataobjects_mock()
        for key, model_base in builder.models():
            test_results = []
            test_trials = []
            for test_fold_index, data_test in enumerate(data_test_list):
                model_class = resolve_model_class(key, base.base_namespace, here)

                model_base_mock = model_base
                model_base_mock = _reset_verbose_option(model_base_mock)

                outcome = runner.run(data_test, base.base_namespace, model_base_mock, model_class, test_fold_index)

                test_results.append(outcome.best_eval)
                if outcome.trials is not None:
                    test_trials.append(outcome.trials)

            min_val = np.argmin([i["loss"] for i in test_results])

            res_handler.add_oneshot_recommender(**test_results[min_val])

            if isinstance(model_base, tuple):
                res_handler.add_trials(test_trials[min_val])
        logger.info("End config test without issues")
    base.base_namespace.config_test = False


def _configure_torch_device(base_namespace, logger=None):
    requested = getattr(base_namespace, "device", None) or getattr(base_namespace, "torch_device", None)
    device = set_device(requested)
    if logger is not None:
        logger.info("Torch device selected", extra={"context": {"device": str(device)}})


if __name__ == '__main__':
    run_experiment("./config/config.yml")
