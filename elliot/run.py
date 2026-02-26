"""
Module description:

"""

__version__ = '0.3.1'

from typing import List, Optional
import copy
import os
from pathlib import Path
import numpy as np

from elliot.dataset import DataSetLoader, build_mock_dataset
from elliot.namespace import build_namespace, ExperimentConfig
from elliot.hyperoptimization import run_hyperopt, run_single
from elliot.result_handler import ResultHandler, attach_test_fold_stats
from elliot.utils import logging as logging_project
from elliot.utils import set_device
from elliot.utils import wandb_logger


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

    mode = _setup_wandb(config, logger, config_path)
    wandb_logger.init_tracking(mode, config, logger)

    _configure_torch_device(config, logger)

    if config.version != __version__:
        logger.error(f'Your config file uses a different version of Elliot! '
                     f'In different versions of Elliot the results may slightly change due to progressive improvement! '
                     f'Some features could be deprecated! Download latest version at this link '
                     f'https://github.com/sisinflab/elliot/releases')
        raise Exception(
            'Version mismatch! In different versions of Elliot the results may slightly change due to progressive improvement!')

    try:
        logger.info("Start experiment")

        dataset_loader = DataSetLoader(config=config)
        data_test_list = dataset_loader.build()

        res_handler = ResultHandler(config=config)

        all_trials = {}

        for model_name, model_config in config.models.items():
            wandb_logger.start_model_run(config, model_name, logger)
            test_results = []
            test_trials = []
            all_trials[model_name] = []

            try:
                for test_fold_index, data_test in enumerate(data_test_list):
                    logging_project.prepare_logger(model_name, config.path_log_folder)

                    is_proxy = model_name.startswith("ProxyRecommender")
                    if is_proxy:
                        logger.info(f"Evaluation begun for {model_name}\n")
                        outcome = run_single(
                            data_test=data_test,
                            config=config,
                            model_config=model_config,
                            model_name=model_name,
                            test_fold_index=test_fold_index
                        )
                        logger.info(f"Evaluation ended for {model_name}")
                    else:
                        logger.info(f"Tuning begun for {model_name}\n")
                        outcome = run_hyperopt(
                            data_test=data_test,
                            config=config,
                            model_config=model_config,
                            model_name=model_name,
                            test_fold_index=test_fold_index
                        )
                        logger.info(f"Tuning ended for {model_name}")
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
                        all_trials[model_name].append(outcome.all_trial_results)

                    logger.info(f"Loss:\t{best_model_loss}")
                    logger.info(f"Best Model params:\t{best_model_params}")
                    logger.info(f"Best Model results:\t{best_model_results}")

                # Migliore sui test, aggiunta a performance totali
                min_val = np.argmin([i["loss"] for i in test_results])
                best_eval = test_results[min_val]

                results_config = config.results
                if results_config.save_fold_stats:
                    attach_test_fold_stats(best_eval, test_results)

                res_handler.add_oneshot_recommender(**best_eval)
                wandb_logger.collect_best_model_result(model_name, best_eval, selected_test_fold=min_val + 1)

                if test_trials:
                    res_handler.add_trials(test_trials[min_val], name=model_name)
                    all_trials[model_name] = all_trials[model_name][min_val]
            finally:
                wandb_logger.finish_model_run(logger)

        res_handler.save_outputs()
        wandb_logger.log_summary_table(config, logger)

        logger.info("End experiment")
    finally:
        wandb_logger.finish(logger)
    # TODO: check before to push only this feature!
    # logger.info("Start Post-Hoc scripts")

    # spec = importlib.util.spec_from_file_location("post_hoc", path.relpath(base.base_namespace.external_posthoc_path))
    # post_hoc = importlib.util.module_from_spec(spec)
    # sys.modules[spec.name] = post_hoc
    # spec.loader.exec_module(post_hoc)
    # post_hoc.run(data_test_list, all_trials)

    # logger.info("End Post-Hoc scripts")


def config_test(config: ExperimentConfig):
    logging_project.init(config.path_logger_config, config.path_log_folder)
    logger = logging_project.get_logger("__main__")

    _configure_torch_device(config, logger)

    logger.info("Start config test")

    data_test_list = build_mock_dataset(config)

    res_handler = ResultHandler(config=config)

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


def _load_dotenv(dotenv_path: Path):
    if not dotenv_path.is_file():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)


def _setup_wandb(config, logger=None, config_path: str = ""):
    wandb_cfg = getattr(config, "wandb", None)
    mode = getattr(wandb_cfg, "mode", "disabled") if wandb_cfg is not None else "disabled"

    if mode == "disabled":
        if logger is not None:
            logger.info("W&B disabled by configuration", extra={"context": {"mode": mode}})
        return mode


    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B mode is enabled but `wandb` is not installed."
            "Install it with `pip install wandb`."
        ) from exc

    project = getattr(wandb_cfg, "project", None)

    if mode == "offline":
        if logger:
            logger.info(
                "W&B offline mode enabled",
                extra={"context": {"mode": mode, "project": project}},
            )

    if mode == "online":

        api_key = os.environ.get("WANDB_API_KEY")

        # Optional local fallback for developer workflows:
        # load .env only if the API key is not already present in the environment.
        if not api_key:
            dotenv_candidates = [Path.cwd() / ".env"]
            if config_path:
                dotenv_candidates.append(Path(config_path).resolve().parent / ".env")

            for dotenv_path in dotenv_candidates:
                _load_dotenv(dotenv_path)

            api_key = os.environ.get("WANDB_API_KEY")

        if not api_key:
            raise RuntimeError(
                "W&B online mode requires WANDB_API_KEY in environment variables."
            )

        try:
            login = wandb.login(key=api_key)
        except Exception as exc:
            raise RuntimeError("W&B login failed.") from exc

        if not login:
            raise RuntimeError("W&B login failed with provided WANDB_API_KEY.")

        if logger:
            logger.info(
                "W&B online login successful",
                extra={"context": {"mode": mode, "project": project}},
            )

    return mode


if __name__ == '__main__':
    run_experiment("./config/config.yml")
