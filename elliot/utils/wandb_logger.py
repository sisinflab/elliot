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
from typing import Any, Optional


@dataclass
class _WandBState:
    enabled: bool = False
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
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        return str(value)


def _is_configured(config) -> bool:
    wandb_cfg = getattr(config, "wandb", None)
    if wandb_cfg is None:
        return False
    return bool(getattr(wandb_cfg, "project", None) and getattr(wandb_cfg, "api_key", None))


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _base_name(config) -> str:
    custom_name = getattr(config.wandb, "run_name", None)
    if isinstance(custom_name, str):
        custom_name = custom_name.strip() or None
    return custom_name or f"elliot-{config.dataset}"


def _setup_metrics(wandb):
    wandb.define_metric("search/global_step")
    wandb.define_metric("search/*", step_metric="search/global_step")


def init_tracking(config, logger=None) -> bool:
    if not _is_configured(config):
        return False

    if STATE.enabled:
        return True

    STATE.enabled = True
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
            "Weights & Biases tracking enabled",
            extra={"context": {"project": STATE.project, "group": STATE.experiment_group}},
        )

    return True


def start_model_run(config, model_name: str, logger=None):
    if not STATE.enabled:
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
    )
    _setup_metrics(wandb)

    STATE.run_name = run_name
    STATE.active_model = model_name
    STATE.global_step = 0

    if logger is not None:
        logger.info(
            "Weights & Biases model run initialized",
            extra={
                "context": {
                    "project": STATE.project,
                    "group": STATE.experiment_group,
                    "run_name": run_name,
                    "model": model_name,
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
    if not STATE.enabled or STATE.run is None:
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
    if not STATE.enabled:
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
    if not STATE.enabled or not STATE.summary_rows:
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
    if not STATE.enabled or STATE.run is None:
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
    if not STATE.enabled:
        return

    if STATE.run is not None:
        import wandb
        wandb.finish()

    if logger is not None:
        logger.info(
            "Weights & Biases tracking closed",
            extra={"context": {"project": STATE.project, "group": STATE.experiment_group}},
        )

    STATE.enabled = False
    STATE.project = None
    STATE.experiment_group = None
    STATE.run = None
    STATE.run_name = None
    STATE.active_model = None
    STATE.global_step = 0
    STATE.summary_rows = None
    STATE.summary_metric_keys = None
