import os
import sys
import importlib
import torch

from elliot.utils.enums import ModelType
from elliot.utils.folder import parent_dir, path_relative

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


def get_model(model_name: str, config):
    if model_name.startswith("external."):
        spec = importlib.util.spec_from_file_location(
            "external",
            path_relative(
                config.external_models_path,
                parent_dir(parent_dir(__file__))
            )
        )
        external = importlib.util.module_from_spec(spec)
        external.backend = config.backend
        sys.modules[spec.name] = external
        spec.loader.exec_module(external)
        model_class = getattr(importlib.import_module("external"), model_name.split(".", 1)[1])
    elif model_name.startswith("Proxy"):
        model_class = getattr(importlib.import_module("elliot.recommender"), "ProxyRecommender")
    else:
        model_class = getattr(importlib.import_module("elliot.recommender"), model_name)

    return model_class


def get_trainer(model_class):
    match model_class.type:
        case ModelType.BASE:
            trainer_name = "Trainer"

        case ModelType.TRADITIONAL:
            trainer_name = "TraditionalTrainer"

        case ModelType.GENERAL:
            trainer_name = "GeneralTrainer"

        case _:
            raise ValueError(f"Unknown model type '{model_class.type}'")

    trainer_class = getattr(importlib.import_module("elliot.recommender"), trainer_name)

    return trainer_class


def split_metric(metric: str):
    split = metric.split("@")
    metric_name = split[0]
    top_k = split[1] if len(split) > 1 else ""
    top_k = int(top_k) if top_k else None
    return metric_name, top_k

#
# def center_data(
#     R: Union[csr_matrix, csc_matrix],
#     axis: int = 0,
#     copy: bool = True
# ) -> csr_matrix:
#
#     if axis == 0:
#         M = R.tocsc(copy=copy)
#     else:
#         M = R.tocsr(copy=copy)
#
#     sums = np.asarray(M.sum(axis=axis)).ravel()
#     nnz = np.diff(M.indptr)
#
#     means = np.zeros_like(sums)
#     mask = nnz > 0
#     means[mask] = sums[mask] / nnz[mask]
#
#     expanded_means = np.repeat(means, nnz)
#     M.data -= expanded_means
#
#     return M.tocsr()
