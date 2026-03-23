from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch


_SUPPORTED_METRICS = (
    "Precision",
    "Recall",
    "HR",
    "MRR",
    "MAP",
    "MAR",
    "F1",
    "nDCG",
)
_LOWER_TO_CANONICAL = {name.lower(): name for name in _SUPPORTED_METRICS}


@dataclass(frozen=True)
class AcceleratedMetricsResult:
    results: Dict[str, float]
    user_results: Dict[str, Dict[Any, float]]
    users: int
    device: str


def supported_metric_names() -> List[str]:
    return list(_SUPPORTED_METRICS)


def canonical_metric_name(name: str) -> Optional[str]:
    return _LOWER_TO_CANONICAL.get(str(name).lower())


def is_supported_metric(name: str) -> bool:
    return canonical_metric_name(name) is not None


def _resolve_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        name = str(requested).strip().lower()
        if name in {"gpu", "cuda"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name in {"mps", "apple", "metal"}:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if name in {"cpu"}:
            return torch.device("cpu")
        return torch.device(name)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_supported(metric_names: Iterable[str]) -> List[str]:
    canonical = []
    for metric in metric_names:
        parsed = canonical_metric_name(metric)
        if parsed is None:
            raise ValueError(f"Unsupported accelerated metric '{metric}'")
        if parsed not in canonical:
            canonical.append(parsed)
    return canonical


def _empty_result(
    metrics: List[str],
    users: List[Any],
    device: torch.device,
    return_user_metrics: bool,
) -> AcceleratedMetricsResult:
    results = {metric: float("nan") for metric in metrics}
    user_results = {metric: {} for metric in metrics} if return_user_metrics else {}
    return AcceleratedMetricsResult(
        results=results,
        user_results=user_results,
        users=len(users),
        device=str(device),
    )


def compute_accelerated_metrics(
    recommendations: Dict[Any, List[tuple]],
    test_data: Dict[Any, Dict[Any, float]],
    cutoff: int,
    relevance_threshold: float,
    metric_names: Iterable[str],
    device: Optional[str] = None,
    return_user_metrics: bool = False,
) -> AcceleratedMetricsResult:
    if cutoff <= 0:
        raise ValueError("Accelerated metrics require cutoff > 0.")

    metrics = _ensure_supported(metric_names)
    target_device = _resolve_device(device)
    dtype = torch.float32 if target_device.type in {"cuda", "mps"} else torch.float64

    active_users = []
    hits_rows = []
    gains_rows = []
    ideal_rows = []
    rel_counts = []

    for user, user_recs in recommendations.items():
        test_items = test_data.get(user)
        if not test_items:
            continue

        rel_gains = {
            item: float(2 ** (score - relevance_threshold + 1) - 1)
            for item, score in test_items.items()
            if score >= relevance_threshold
        }
        if not rel_gains:
            continue

        rec_items = [item for item, _ in user_recs[:cutoff]]
        hit_row = [1.0 if item in rel_gains else 0.0 for item in rec_items]
        gain_row = [rel_gains.get(item, 0.0) for item in rec_items]

        if len(rec_items) < cutoff:
            pad = cutoff - len(rec_items)
            hit_row.extend([0.0] * pad)
            gain_row.extend([0.0] * pad)

        ideal = sorted(rel_gains.values(), reverse=True)[:cutoff]
        if len(ideal) < cutoff:
            ideal.extend([0.0] * (cutoff - len(ideal)))

        active_users.append(user)
        hits_rows.append(hit_row)
        gains_rows.append(gain_row)
        ideal_rows.append(ideal)
        rel_counts.append(float(len(rel_gains)))

    if not active_users:
        return _empty_result(metrics, active_users, target_device, return_user_metrics)

    with torch.inference_mode():
        hits = torch.tensor(hits_rows, dtype=dtype, device=target_device)
        gains = torch.tensor(gains_rows, dtype=dtype, device=target_device)
        ideal_gains = torch.tensor(ideal_rows, dtype=dtype, device=target_device)
        rel_count = torch.tensor(rel_counts, dtype=dtype, device=target_device)

        ranks = torch.arange(1, cutoff + 1, dtype=dtype, device=target_device)
        discounts = 1.0 / torch.log2(torch.arange(2, cutoff + 2, dtype=dtype, device=target_device))

        cumulative_hits = torch.cumsum(hits, dim=1)
        hits_at_k = cumulative_hits[:, -1]

        precision = hits_at_k / float(cutoff)
        recall = hits_at_k / rel_count
        hr = (hits_at_k > 0).to(dtype)

        precision_prefix = cumulative_hits / ranks.unsqueeze(0)
        map_metric = precision_prefix.mean(dim=1)
        mar_metric = (cumulative_hits / rel_count.unsqueeze(1)).mean(dim=1)

        first_hit_idx = torch.argmax((hits > 0).to(torch.int64), dim=1) + 1
        has_hit = torch.any(hits > 0, dim=1)
        mrr = torch.where(has_hit, 1.0 / first_hit_idx.to(dtype), torch.zeros_like(precision))

        dcg = torch.sum(gains * discounts.unsqueeze(0), dim=1)
        idcg = torch.sum(ideal_gains * discounts.unsqueeze(0), dim=1)
        ndcg = torch.where(
            dcg > 0,
            dcg / torch.clamp(idcg, min=torch.finfo(dtype).eps),
            torch.zeros_like(dcg),
        )

        f1_den = precision + recall
        f1 = torch.where(f1_den > 0, 2.0 * precision * recall / f1_den, torch.zeros_like(f1_den))

        tensor_by_metric = {
            "Precision": precision,
            "Recall": recall,
            "HR": hr,
            "MRR": mrr,
            "MAP": map_metric,
            "MAR": mar_metric,
            "F1": f1,
            "nDCG": ndcg,
        }

        results = {
            metric: float(torch.mean(tensor_by_metric[metric]).detach().cpu().item())
            for metric in metrics
        }

        user_results: Dict[str, Dict[Any, float]] = {}
        if return_user_metrics:
            for metric in metrics:
                values = tensor_by_metric[metric].detach().cpu().tolist()
                user_results[metric] = {
                    user: float(value)
                    for user, value in zip(active_users, values)
                }

    return AcceleratedMetricsResult(
        results=results,
        user_results=user_results,
        users=len(active_users),
        device=str(target_device),
    )
