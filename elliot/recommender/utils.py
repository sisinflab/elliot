import os
import torch

from elliot.recommender.base_trainer import Trainer, TraditionalTrainer, GeneralTrainer
from elliot.utils.enums import ModelType


_DEVICE = None


def _auto_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_device_name(value):
    if value is None:
        return None
    name = str(value).strip().lower()
    if name in {"auto", ""}:
        return None
    if name in {"gpu", "cuda"}:
        return "cuda"
    if name in {"mps", "apple", "metal"}:
        return "mps"
    if name in {"cpu"}:
        return "cpu"
    return name


def set_device(requested=None):
    global _DEVICE
    env_value = os.environ.get("ELLIOT_DEVICE", None)
    name = _normalize_device_name(requested if requested is not None else env_value)
    if name is None:
        _DEVICE = _auto_device()
        return _DEVICE
    if name == "cuda":
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _DEVICE
    if name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _DEVICE = torch.device("mps")
        else:
            _DEVICE = torch.device("cpu")
        return _DEVICE
    _DEVICE = torch.device(name)
    return _DEVICE


def get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = set_device(None)
    return _DEVICE


def get_model(data, config, params, model_class):
    match model_class.type:
        case ModelType.BASE:
            trainer = Trainer

        case ModelType.TRADITIONAL:
            trainer = TraditionalTrainer

        case ModelType.GENERAL:
            trainer = GeneralTrainer

        case _:
            raise ValueError(f"Unknown model type '{model_class.type}'")

    return trainer(data, config, params, model_class)
