"""
Optional Weights & Biases tracking helpers.

Behavior:
- if W&B is configured, create one run per hyperopt trial
- log trial trends and metadata in the corresponding trial run
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Literal


@dataclass
class _WandBState:
    mode: Optional[Literal["online", "offline", "disabled"]] = None
    project: Optional[str] = None
    experiment_group: Optional[str] = None
    active_model: Optional[str] = None
    summary_rows: list = None
    summary_metric_keys: set = None


STATE = _WandBState()


def _sanitize(value: Any) -> Any:
    """Convert values to W&B-safe primitives for logging.

    Primitive values are returned unchanged. Non-primitive values are
    serialized to JSON when possible, with a final string fallback to avoid
    runtime failures during logging.

    Args:
        value (Any): Value to sanitize before logging.

    Returns:
        Any: A W&B-compatible value (primitive or stringified representation).
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _timestamp() -> str:
    """Return a UTC timestamp string used to build unique W&B names.

    Returns:
        str: Timestamp in the format YYYYMMDD-HHMMSS.
    """

    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _base_name(config) -> str:
    """Build the base name used for W&B run and group naming.

    If a custom run prefix/name is provided in the W&B config, it is used;
    otherwise the default pattern ``elliot-{dataset}`` is applied.

    Args:
        config (ExperimentConfig): Experiment configuration namespace.

    Returns:
        str: Base name for W&B run/group identifiers.
    """

    custom_name = getattr(config.wandb, "run_prefix", None)
    if isinstance(custom_name, str):
        custom_name = custom_name.strip() or None
    return custom_name or f"elliot-{config.dataset}"


def _slugify(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.=@")
    return "".join(ch if ch in allowed else "-" for ch in value)


def _trial_hparams_label(hyperparams: Optional[dict], max_len: int = 120) -> str:
    if not isinstance(hyperparams, dict) or not hyperparams:
        return "default"
    parts = []
    for key in sorted(hyperparams.keys()):
        value = hyperparams.get(key)
        parts.append(f"{key}={value}")
    label = "__".join(parts)
    label = _slugify(label)
    if len(label) > max_len:
        return label[:max_len]
    return label


def _setup_metrics(wandb):
    """Reserved hook for metric setup."""
    return None


def _collect_system_metrics() -> dict:
    metrics = {}

    # 1) Cross-platform baseline via psutil (macOS/Linux/Windows).
    try:
        import psutil

        process = psutil.Process(os.getpid())
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()

        metrics["system.cpu"] = float(psutil.cpu_percent(interval=None))
        metrics["system.memory"] = float(vm.percent)
        metrics["system.memory.usedGB"] = float(vm.used / (1024 ** 3))
        metrics["system.memory.availableGB"] = float(vm.available / (1024 ** 3))
        metrics["system.swap"] = float(swap.percent)
        metrics["proc.memory.rssMB"] = float(process.memory_info().rss / (1024 ** 2))
        metrics["proc.cpu"] = float(process.cpu_percent(interval=None))
        metrics["proc.cpu.threads"] = int(process.num_threads())
        metrics["system.disk"] = float(psutil.disk_usage("/").percent)
        return metrics
    except Exception:
        pass

    # 2) Fallback when psutil is unavailable.
    try:
        cpu_count = os.cpu_count() or 1
        load_avg_1m = os.getloadavg()[0]
        metrics["system.cpu"] = float((load_avg_1m / cpu_count) * 100.0)
    except Exception:
        pass

    try:
        total_pages = os.sysconf("SC_PHYS_PAGES")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_bytes = float(total_pages * page_size)
        avail_bytes = float(avail_pages * page_size)
        used_bytes = max(total_bytes - avail_bytes, 0.0)
        metrics["system.memory"] = float((used_bytes / total_bytes) * 100.0) if total_bytes else 0.0
        metrics["system.memory.usedGB"] = used_bytes / (1024 ** 3)
        metrics["system.memory.availableGB"] = avail_bytes / (1024 ** 3)
    except Exception:
        pass

    # 3) Optional NVIDIA GPU telemetry.
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            for gpu_id in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics[f"system.gpu.{gpu_id}.gpu"] = float(util.gpu)
                metrics[f"system.gpu.{gpu_id}.memory"] = float(util.memory)
                metrics[f"system.gpu.{gpu_id}.memoryAllocatedGB"] = float(mem.used / (1024 ** 3))
                metrics[f"system.gpu.{gpu_id}.memoryTotalGB"] = float(mem.total / (1024 ** 3))
                metrics[f"system.gpu.{gpu_id}.temp"] = float(
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                )
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    # 4) Optional CUDA memory from torch (works even without pynvml).
    try:
        import torch

        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                metrics[f"system.gpu.{gpu_id}.memoryAllocatedGB"] = float(
                    torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
                )
                metrics[f"system.gpu.{gpu_id}.memoryReservedGB"] = float(
                    torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
                )
    except Exception:
        pass

    # 5) Apple Silicon hint for runs on macOS where detailed GPU telemetry is unavailable.
    try:
        if platform.system().lower() == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
            metrics["system.gpu.0.backend"] = "mps"
    except Exception:
        pass

    return metrics


def init_tracking(mode, config, logger=None):
    """Initialize W&B tracking state for the current experiment.

    This function sets the shared tracking context (project, group, and
    runtime state) only once per process. If mode is "disabled", tracking
    remains inactive and no W&B runs are created.

    Args:
        mode (Literal["online", "offline", "disabled"]): W&B execution mode
            resolved during setup.
        config (ExperimentConfig): Experiment configuration namespace.
        logger (logging.Logger, optional): Elliot logger used for status logs.
    """

    # if already enabled exit
    if STATE.mode is not None:
        return

    STATE.mode = mode

    if mode not in ["online", "offline", "disabled"]:
        raise ValueError(f"Invalid mode: {mode}")

    if STATE.mode == "disabled":
        return

    if mode in {"online", "offline"}:
        STATE.project = config.wandb.project
        STATE.experiment_group = f"{_base_name(config)}-{_timestamp()}"
        STATE.active_model = None
        STATE.summary_rows = []
        STATE.summary_metric_keys = set()

        if logger is not None:
            logger.info(
                f"Weights & Biases tracking enabled in mode {STATE.mode}",
                extra={"context": {"project": STATE.project, "group": STATE.experiment_group}},
            )

        return


def start_model_run(config, model_name: str, logger=None):
    """Track currently active model in W&B state.

    With trial-level logging, no run is opened at model scope. The function
    stores model metadata for compatibility with the existing call sites.

    Args:
        config (ExperimentConfig): Experiment configuration namespace.
        model_name (str): Name of the model currently being processed.
        logger (logging.Logger, optional): Elliot logger used for status logs.
    """

    if STATE.mode not in ["online", "offline"]:
        return

    STATE.active_model = model_name

    if logger is not None:
        logger.info(
            "Weights & Biases model context initialized",
            extra={
                "context": {
                    "project": STATE.project,
                    "group": STATE.experiment_group,
                    "model": STATE.active_model,
                    "mode": STATE.mode,
                }
            },
        )


def log_hyperopt_trial(
    *,
    config,
    model_name: str,
    test_fold_index: int,
    trial_index: int,
    hyperparams: Optional[dict],
    objective: Optional[dict],
    payload: Optional[dict],
):
    """Log a single hyperparameter-search trial to a dedicated W&B run.

    The function records only training and validation trends for a trial run.

    Logging is skipped when tracking mode is disabled.
    """

    if STATE.mode not in {"online", "offline"}:
        return

    metric = objective.get("metric") if objective else None
    k = objective.get("k") if objective else None
    metric_label = f"{metric}@{k}" if metric is not None and k is not None else None

    import wandb

    run_name = (
        f"{_base_name(config)}-{model_name}-fold{int(test_fold_index)+1}-trial{int(trial_index)}"
        f"-{_trial_hparams_label(hyperparams)}-{_timestamp()}"
    )
    run_config = {
        "dataset": config.dataset,
        "top_k": config.top_k,
        "random_seed": config.random_seed,
        "model_name": model_name,
        "test_fold": int(test_fold_index) + 1,
        "trial_index": int(trial_index),
    }

    run = wandb.init(
        project=STATE.project,
        group=STATE.experiment_group,
        name=run_name,
        config=run_config,
        tags=[model_name, "trial"],
        job_type="trial",
        reinit=True,
        mode=STATE.mode
    )

    _setup_metrics(wandb)

    try:
        trend = payload.get("trend") if payload else None
        if isinstance(trend, dict):
            epochs = trend.get("epochs", [])
            train_losses = trend.get("train_loss", [])
            val_epochs = trend.get("val_epochs", [])
            val_losses = trend.get("val_loss", [])
            val_metrics = trend.get("val_metric", [])
            trend_metric_name = trend.get("val_metric_name") or metric_label

            train_map = {}
            for idx in range(max(len(epochs), len(train_losses))):
                step = int(epochs[idx]) if idx < len(epochs) and epochs[idx] is not None else idx + 1
                if idx < len(train_losses) and train_losses[idx] is not None:
                    train_map[step] = _sanitize(train_losses[idx])

            val_loss_map = {}
            val_metric_map = {}
            for idx in range(max(len(val_epochs), len(val_losses), len(val_metrics))):
                step = int(val_epochs[idx]) if idx < len(val_epochs) and val_epochs[idx] is not None else idx + 1
                if idx < len(val_losses) and val_losses[idx] is not None:
                    val_loss_map[step] = _sanitize(val_losses[idx])
                if idx < len(val_metrics) and val_metrics[idx] is not None:
                    val_metric_map[step] = _sanitize(val_metrics[idx])

            all_steps = sorted(set(train_map.keys()) | set(val_loss_map.keys()) | set(val_metric_map.keys()))
            for step in all_steps:
                point = {}
                if step in train_map:
                    point["train/loss"] = train_map[step]
                if step in val_loss_map:
                    point["validation/loss"] = val_loss_map[step]
                if step in val_metric_map:
                    if trend_metric_name:
                        point[f"validation/{trend_metric_name}"] = val_metric_map[step]
                    else:
                        point["validation/metric"] = val_metric_map[step]

                point.update(_collect_system_metrics())
                if point:
                    wandb.log(point, step=step)
    finally:
        run.finish()


def _flatten_test_results(test_results: Optional[dict]) -> dict:
    """Flatten nested test metrics into ``metric@cutoff`` key-value pairs.

    Args:
        test_results (dict, optional): Nested test results as
            ``{cutoff: {metric_name: value}}``.

    Returns:
        dict: Flattened mapping as ``{f"{metric}@{cutoff}": value}``.
    """
    out = {}
    if not isinstance(test_results, dict):
        return out
    for cutoff, metrics in test_results.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, value in metrics.items():
            out[f"{metric_name}@{cutoff}"] = value
    return out


def _params_label(params: Optional[dict]) -> str:
    """Build a compact, human-readable label from selected model parameters.

    Internal/meta keys are skipped and only primitive values are included.
    The final string is truncated to avoid excessively long table row names.

    Args:
        params (dict, optional): Model parameter dictionary.

    Returns:
        str: Compact parameter label string.
    """
    if not isinstance(params, dict):
        return ""
    parts = []
    for key in sorted(params.keys()):
        if key in {"meta", "name", "best_iteration"}:
            continue
        value = params.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            parts.append(f"{key}={value}")
    label = ", ".join(parts)
    if len(label) > 180:
        return label[:177] + "..."
    return label


def collect_best_model_result(model_name: str, best_eval: Optional[dict], selected_test_fold: int):
    """Collect the best evaluation result of a model for final summary logging.

    The function stores one normalized row per model in in-memory summary
    buffers, to be later exported as a W&B comparison table.

    Collection is skipped when tracking mode is disabled or when the provided
    evaluation payload is invalid.

    Args:
        model_name (str): Name of the evaluated model.
        best_eval (dict, optional): Best evaluation payload returned by Elliot.
        selected_test_fold (int): 1-based index of the selected best test fold.
    """

    if STATE.mode not in {"online", "offline"}:
        return
    if not isinstance(best_eval, dict):
        return

    params = best_eval.get("params", {})
    flattened_metrics = _flatten_test_results(best_eval.get("test_results", {}))
    STATE.summary_metric_keys.update(flattened_metrics.keys())

    params_label = _params_label(params)
    row_name = model_name if not params_label else f"{model_name} | {params_label}"

    STATE.summary_rows.append(
        {
            "row_name": row_name,
            "model_name": model_name,
            "selected_test_fold": int(selected_test_fold),
            "evaluation_source": "elliot.test_results",
            "best_params": _sanitize(params),
            "metrics": flattened_metrics,
        }
    )


def log_summary_table(config, logger=None):
    """Log a final W&B comparison table with best test results per model.

    A dedicated summary run is created in the current experiment group, where
    a ``comparison/test_results_table`` artifact is logged with one row per
    selected best model result.

    Logging is skipped when tracking mode is disabled or when no summary rows
    were collected.

    Args:
        config (ExperimentConfig): Experiment configuration namespace.
        logger (logging.Logger, optional): Elliot logger used for status logs.
    """

    if STATE.mode not in {'online', 'offline'} or not STATE.summary_rows:
        return

    import wandb

    run_name = f"{_base_name(config)}-summary-{_timestamp()}"
    run = wandb.init(
        project=STATE.project,
        group=STATE.experiment_group,
        name=run_name,
        config={
            "dataset": config.dataset,
            "top_k": config.top_k,
            "random_seed": config.random_seed,
            "summary_type": "test_results_comparison",
        },
        tags=["summary", "comparison"],
        job_type="summary",
        reinit=True,
        mode=STATE.mode
    )
    try:
        metric_columns = sorted(STATE.summary_metric_keys)
        columns = ["row_name", "model_name", "selected_test_fold", "evaluation_source", "best_params"] + metric_columns
        table = wandb.Table(columns=columns)

        for row in STATE.summary_rows:
            values = [
                row["row_name"],
                row["model_name"],
                row["selected_test_fold"],
                row["evaluation_source"],
                row["best_params"],
            ]
            metric_values = [row["metrics"].get(metric_name) for metric_name in metric_columns]
            table.add_data(*(values + metric_values))

        wandb.log({"comparison/test_results_table": table})
        if logger is not None:
            logger.info(
                "Weights & Biases comparison table logged",
                extra={
                    "context": {
                        "project": STATE.project,
                        "group": STATE.experiment_group,
                        "run_name": run_name,
                        "rows": len(STATE.summary_rows),
                        "metric_columns": len(metric_columns),
                    }
                },
            )
    finally:
        run.finish()


def finish_model_run(logger=None):
    """Clear active model metadata from W&B state.

    With trial-level logging there is no open model-level run to close.

    Args:
        logger (logging.Logger, optional): Elliot logger used for status logs.
    """

    if STATE.mode not in {"online","offline"}:
        return

    if logger is not None:
        logger.info(
            "Weights & Biases model context cleared",
            extra={"context": {"project": STATE.project, "model": STATE.active_model}},
        )

    STATE.active_model = None


def finish(logger=None):
    """Close W&B tracking for the experiment and reset global tracking state.

    If a model run is still active, it is closed before resetting state.

    Args:
        logger (logging.Logger, optional): Elliot logger used for status logs.
    """

    if STATE.mode not in {"online","offline"}:
        return

    if logger is not None:
        logger.info(
            "Weights & Biases tracking closed",
            extra={"context": {"project": STATE.project, "group": STATE.experiment_group}},
        )

    STATE.mode = None
    STATE.project = None
    STATE.experiment_group = None
    STATE.active_model = None
    STATE.summary_rows = []
    STATE.summary_metric_keys = set()