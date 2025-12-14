"""
Hydra-backed loader for Elliot experiment configs.
"""

import os
from typing import Any, Dict, List, Optional

from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str, overrides: Optional[List[str]] = None, job_name: str = "elliot") -> Dict[str, Any]:
    """
    Load and resolve a configuration file using Hydra, with optional override strings.

    :param config_path: Absolute or relative path to the YAML configuration.
    :param overrides: Optional list of Hydra-style override strings.
    :param job_name: Hydra job name to use for isolation (no side effects on caller's Hydra stack).
    :return: A plain dictionary representation of the resolved config.
    """
    cfg = compose_config(config_path, overrides=overrides or [], job_name=job_name)
    return OmegaConf.to_container(cfg, resolve=True)


def compose_config(config_path: str, overrides: List[str], job_name: str) -> DictConfig:
    config_dir = os.path.abspath(os.path.dirname(config_path) or ".")
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    try:
        with initialize_config_dir(
            version_base="1.3",
            config_dir=config_dir,
            job_name=job_name,
        ):
            return compose(config_name=config_name, overrides=overrides, return_hydra_config=False)
    except HydraException as e:
        raise ValueError(f"Unable to load configuration '{config_path}' via Hydra") from e
