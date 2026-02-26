"""
Optional Weights & Biases tracking helpers.

Behavior:
- if W&B is configured, create one run per model
- log all that model's hyperopt trials into that run
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Literal


@dataclass
class _WandBState:
    mode: Optional[Literal["online", "offline", "disabled"]] = None
    project: Optional[str] = None
    experiment_group: Optional[str] = None
    run: Optional[object] = None
    run_name: Optional[str] = None
    active_model: Optional[str] = None
    global_step: int = 0
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


def _setup_metrics(wandb):
    """Register W&B metric definitions for hyperparameter search logs.

    This configures ``search/global_step`` as the shared x-axis and binds
    all metrics under ``search/*`` to that step, so trial-level charts are
    aligned across the run.

    Args:
        wandb: Imported Weights & Biases module.
    """

    wandb.define_metric("search/global_step")
    wandb.define_metric("search/*", step_metric="search/global_step")


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
        STATE.run = None
        STATE.run_name = None
        STATE.active_model = None
        STATE.global_step = 0
        STATE.summary_rows = []
        STATE.summary_metric_keys = set()

        if logger is not None:
            logger.info(
                f"Weights & Biases tracking enabled in mode {STATE.mode}",
                extra={"context": {"project": STATE.project, "group": STATE.experiment_group}},
            )

        return


def start_model_run(config, model_name: str, logger=None):
    """Start a W&B run for a single model inside the current experiment group.

    The run is created only when tracking mode is "online" or "offline".
    If another run is still active, it is closed before creating the new one.

    Args:
        config (ExperimentConfig): Experiment configuration namespace.
        model_name (str): Name of the model currently being processed.
        logger (logging.Logger, optional): Elliot logger used for status logs.
    """

    if STATE.mode not in ["online", "offline"]:
        return

    import wandb

    if STATE.run is not None:
        finish_model_run(logger)

    run_name = f"{_base_name(config)}-{model_name}-{_timestamp()}"
    run_config = {
        "dataset": config.dataset,
        "top_k": config.top_k,
        "random_seed": config.random_seed,
        "model_name": model_name,
    }

    STATE.run = wandb.init(
        project=STATE.project,
        group=STATE.experiment_group,
        name=run_name,
        config=run_config,
        tags=[model_name],
        reinit=True,
        mode=STATE.mode
    )

    STATE.run_name = run_name
    STATE.active_model = model_name
    STATE.global_step = 0

    _setup_metrics(wandb)

    if logger is not None:
        logger.info(
            "Weights & Biases model run initialized",
            extra={
                "context": {
                    "project": STATE.project,
                    "group": STATE.experiment_group, # forse rimovibile
                    "run_name": STATE.run_name,
                    "model": STATE.active_model,
                    "mode": STATE.mode,
                }
            },
        )


def log_hyperopt_trial(
    *,
    model_name: str,
    test_fold_index: int,
    trial_index: int,
    hyperparams: Optional[dict],
    objective: Optional[dict],
    payload: Optional[dict],
):
    """Log a single hyperparameter-search trial to the active W&B model run.

    The function records trial identifiers, objective metadata, selected
    validation values, loss, and trial hyperparameters under the ``search/*``
    and ``hparams/*`` namespaces.

    Logging is skipped when tracking mode is disabled or no run is active.
    """

    if STATE.mode not in {"online", "offline"} or STATE.run is None:
        return

    STATE.global_step += 1
    safe_model_name = str(model_name).replace("/", "_")
    model_trial_id = f"{model_name}|fold={int(test_fold_index)+1}|trial={int(trial_index)}"

    data = {
        "search/global_step": int(STATE.global_step),
        "search/model_name": model_name,
        "search/test_fold": int(test_fold_index) + 1,
        "search/trial_index": int(trial_index),
        "search/model_trial_id": model_trial_id,
    }

    if payload:
        data["search/loss"] = _sanitize(payload.get("loss"))
        val_metric = _sanitize(payload.get("val_metric"))
        data["search/val_metric"] = val_metric
        data[f"search/val_metric_{safe_model_name}"] = val_metric

    if objective:
        metric = objective.get("metric")
        k = objective.get("k")
        target = objective.get("target")
        direction = objective.get("direction")
        value = objective.get("value")

        metric_label = f"{metric}@{k}" if metric is not None and k is not None else None
        data["search/objective_target"] = _sanitize(target)
        data["search/objective_direction"] = _sanitize(direction)
        data["search/objective_metric"] = _sanitize(metric_label)
        data["search/objective_value"] = _sanitize(value)

    for key, value in (hyperparams or {}).items():
        data[f"hparams/{key}"] = _sanitize(value)

    import wandb
    wandb.log(data)


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
    """Close the currently active W&B model run and reset run-level state.

    If no model run is active, the function is a no-op.

    Args:
        logger (logging.Logger, optional): Elliot logger used for status logs.
    """

    if STATE.mode not in {"online","offline"} or STATE.run is None:
        return

    import wandb
    wandb.finish()

    if logger is not None:
        logger.info(
            "Weights & Biases model run closed",
            extra={"context": {"project": STATE.project, "run_name": STATE.run_name, "model": STATE.active_model}},
        )

    STATE.run = None
    STATE.run_name = None
    STATE.active_model = None
    STATE.global_step = 0


def finish(logger=None):
    """Close W&B tracking for the experiment and reset global tracking state.

    If a model run is still active, it is closed before resetting state.

    Args:
        logger (logging.Logger, optional): Elliot logger used for status logs.
    """

    if STATE.mode not in {"online","offline"}:
        return

    if STATE.run is not None:
        import wandb
        wandb.finish()

    if logger is not None:
        logger.info(
            "Weights & Biases tracking closed",
            extra={"context": {"project": STATE.project, "group": STATE.experiment_group}},
        )

    STATE.mode = None
    STATE.project = None
    STATE.experiment_group = None
    STATE.run = None
    STATE.run_name = None
    STATE.active_model = None
    STATE.global_step = 0
    STATE.summary_rows = []
    STATE.summary_metric_keys = set()
