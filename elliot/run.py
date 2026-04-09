"""
Module description:

"""

__version__ = '0.3.1'

from typing import List, Optional
import copy
import os
from pathlib import Path
import socket
import numpy as np

from elliot.dataset import DataSetLoader, build_mock_dataset
from elliot.namespace import build_namespace, ExperimentConfig
from elliot.hyperoptimization import run_hyperopt, run_single, run_proxy, run_evaluation, requires_hyperopt
from elliot.result_handler import ResultHandler, attach_test_fold_stats
from elliot.utils import logging as logging_project
from elliot.utils import set_device
from elliot.utils import wandb_logger
from elliot.utils.callback import CallbackManager
from elliot.utils.registry import callback_registry

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
    logging_project.init()
    logger = logging_project.get_logger("__main__")

    config = build_namespace(config_path, config_overrides)

    if config.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

    if config.config_test:
        config_test(config)

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

    # Load callbacks
    cb_manager = CallbackManager(callbacks=callback_registry.get_all())

    try:
        logger.info("Start experiment")

        # NOTE: to be discussed
        # Callback on experiment start

        loader = DataSetLoader(config=config)

        # Callback on data loading and filtering
        cb_manager.trigger(
            event_name="on_data_loading_and_filtering",
            data=loader.dataframe
        )

        val_data, main_data = loader.build()

        # Callback on dataset creation
        cb_manager.trigger(
            event_name="on_dataset_creation",
            val_dataset=val_data,
            main_dataset=main_data
        )

        loader.prepare_dataset(val_data, main_data)

        res_handler = ResultHandler(config=config)

        all_trials = {}

        for model_name, model_config in config.models.items():
            wandb_logger.start_model_run(config, model_name, logger)
            test_results = []
            test_trials = []
            all_trials[model_name] = []

            try:
                # Callback on model start
                cb_manager.trigger(
                    event_name="on_model_start",
                    model_name=model_name,
                    model_config=model_config
                )

                for test_fold_index, (val, main) in enumerate(zip(val_data, main_data)):
                    logging_project.prepare_logger(model_name)

                    is_proxy = model_name.startswith("ProxyRecommender")
                    use_hyperopt = requires_hyperopt(model_config)

                    if use_hyperopt:
                        logger.info(f"Tuning begun for {model_name}\n")
                        outcome = run_hyperopt(
                            val_data=val,
                            main_data=main,
                            config=config,
                            model_config=model_config,
                            model_name=model_name,
                            test_fold_index=test_fold_index
                        )
                        logger.info(f"Tuning ended for {model_name}")

                    elif is_proxy:
                        outcome = run_proxy(
                            model_config=model_config,
                            main_data=main,
                            config=config
                        )

                    else:
                        logger.info(f"Training begun for {model_name}\n")
                        outcome = run_single(
                            val_data=val,
                            main_data=main,
                            config=config,
                            model_config=model_config,
                            model_name=model_name,
                            test_fold_index=test_fold_index
                        )
                        logger.info(f"Training ended for {model_name}")

                    model = outcome.best_model
                    results = outcome.results

                    # Callback on training complete
                    cb_manager.trigger(
                        event_name="on_training_complete",
                        model=model,
                        results=results
                    )

                    logger.info(f"Evaluation begun for {model_name}\n")
                    eval_results = run_evaluation(
                        main_data=main,
                        config=config,
                        model=model
                    )
                    logger.info(f"Evaluation ended for {model_name}")

                    # Callback on evaluation complete
                    cb_manager.trigger(
                        event_name="on_evaluation_complete",
                        model=model,
                        results=eval_results
                    )

                    results.update(eval_results)

                    ############################################
                    best_model_loss = results["loss"]
                    best_model_params = results["params"]
                    best_model_results = results["test_results"]
                    ############################################

                    # aggiunta a lista performance test
                    test_results.append(results)

                    if outcome.trials is not None:
                        test_trials.append(outcome.all_trial_results)
                        all_trials[model_name].append(outcome.all_trial_results)

                    logger.info(f"Loss:\t{best_model_loss}")
                    logger.info(f"Best Model params:\t{best_model_params}")
                    logger.info(f"Best Model results:\t{best_model_results}")

                # Callback on model complete
                cb_manager.trigger(
                    event_name="on_model_complete",
                    model_name=model_name,
                    results=test_results,
                    trials=test_trials
                )

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

        # NOTE: to be discussed
        # Callback on experiment complete

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
    logging_project.init()
    logger = logging_project.get_logger("__main__")

    _configure_torch_device(config, logger)

    logger.info("Start config test")

    val_data, main_data = build_mock_dataset(config)

    res_handler = ResultHandler(config=config)

    for model_name, model_config in config.models.items():
        test_results = []
        test_trials = []

        for test_fold_index, (val, main) in enumerate(zip(val_data, main_data)):
            model_config_mock = copy.deepcopy(model_config)
            model_config_mock.meta.verbose = False
            model_config_mock.meta.save_recs = False
            model_config_mock.meta.save_weights = False

            is_proxy = model_name.startswith("ProxyRecommender")
            use_hyperopt = requires_hyperopt(model_config_mock)

            if use_hyperopt:
                outcome = run_hyperopt(
                    val_data=val,
                    main_data=main,
                    config=config,
                    model_config=model_config_mock,
                    model_name=model_name,
                    test_fold_index=test_fold_index
                )
            elif is_proxy:
                outcome = run_proxy(
                    model_config=model_config_mock,
                    main_data=main,
                    config=config
                )
            else:
                outcome = run_single(
                    val_data=val,
                    main_data=main,
                    config=config,
                    model_config=model_config_mock,
                    model_name=model_name,
                    test_fold_index=test_fold_index
                )

            model = outcome.best_model
            results = outcome.results

            eval_results = run_evaluation(
                main_data=main,
                config=config,
                model=model
            )

            results.update(eval_results)

            test_results.append(results)
            if outcome.trials is not None:
                test_trials.append(outcome.all_trial_results)

        min_val = np.argmin([i["loss"] for i in test_results])

        if test_trials:
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


def _has_wandb_online_connectivity(host: str = "api.wandb.ai", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


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
        if not _has_wandb_online_connectivity():
            raise RuntimeError(
                "W&B online mode requires internet connectivity."
                " Unable to reach api.wandb.ai:443."
            )

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
