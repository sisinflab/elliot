import multiprocessing as mp

if mp.current_process().name != "MainProcess":
    # Dataloader worker started via `spawn` (Windows/macOS): this module is re-imported
    # before anything else, so silence import-time noise from third-party libraries
    # that the main process has already shown once.
    import logging
    import os
    import warnings

    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.disable(logging.WARNING)


def load_wandb_api_key(env_path):
    """Set `WANDB_API_KEY` from a `.env` file (`KEY=VALUE` lines), unless it is
    already defined in the environment."""
    import os
    from pathlib import Path

    if os.environ.get("WANDB_API_KEY"):
        return
    env_file = Path(env_path)
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        if key.strip() == "WANDB_API_KEY":
            os.environ["WANDB_API_KEY"] = value.strip().strip("'\"")
            return


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from elliot.run import run_experiment

    load_wandb_api_key(Path(__file__).resolve().parent / ".env")

    parser = argparse.ArgumentParser(description="Run sample main.")
    parser.add_argument(
        '--config', type=str, required=True,
        help='Experiment configuration file name (without .yaml extension)'
    )
    args = parser.parse_args()

    run_experiment(f"config_files/{args.config}.yaml")
