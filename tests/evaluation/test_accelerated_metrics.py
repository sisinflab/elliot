import math
from types import SimpleNamespace

import pytest

from elliot.evaluation.accelerated_metrics import compute_accelerated_metrics
from elliot.evaluation.relevance import Relevance
from elliot.utils.registry import metric_registry

metric_names = ["Precision", "Recall", "HR", "MRR", "MAP", "MAR", "F1", "nDCG"]


def _legacy_results(recommendations, test_data, cutoff, threshold):
    eval_objects = SimpleNamespace(
        cutoff=cutoff,
        relevance=Relevance(test_data, threshold),
        num_items=100,
        data=SimpleNamespace(get_dict=lambda: {}),
    )

    scalar = {}
    users = {}
    for metric_name in metric_names:
        metric = metric_registry.get(
            name=metric_name,
            recommendations=recommendations,
            config=SimpleNamespace(config_test=True),
            params=SimpleNamespace(),
            eval_objects=eval_objects,
        )
        scalar[metric.name] = metric.eval()
        users[metric.name] = metric.eval_user_metric()

    return scalar, users


def test_accelerated_metrics_match_legacy():
    test_data = {
        "u1": {"i1": 1.0, "i2": 3.0, "i7": 0.0},
        "u2": {"i3": 2.0, "i4": 0.0},
        "u3": {"i9": 0.0},
    }
    recommendations = {
        "u1": [("i2", 0.9), ("i5", 0.8), ("i1", 0.7), ("i6", 0.3)],
        "u2": [("i8", 0.9), ("i3", 0.8), ("i6", 0.2)],
        "u3": [("i9", 0.9), ("i10", 0.8)],
    }
    cutoff = 3
    threshold = 1.0

    accelerated = compute_accelerated_metrics(
        recommendations=recommendations,
        test_data=test_data,
        cutoff=cutoff,
        relevance_threshold=threshold,
        metric_names=metric_names,
        device="cpu",
        return_user_metrics=True,
    )
    legacy_scalar, legacy_users = _legacy_results(
        recommendations=recommendations,
        test_data=test_data,
        cutoff=cutoff,
        threshold=threshold,
    )

    for metric_name in metric_names:
        assert math.isclose(
            accelerated.results[metric_name],
            legacy_scalar[metric_name],
            rel_tol=1e-8,
            abs_tol=1e-8,
        )

        assert set(accelerated.user_results[metric_name].keys()) == set(legacy_users[metric_name].keys())
        for user_id in accelerated.user_results[metric_name].keys():
            assert math.isclose(
                accelerated.user_results[metric_name][user_id],
                legacy_users[metric_name][user_id],
                rel_tol=1e-8,
                abs_tol=1e-8,
            )


def test_accelerated_metrics_no_evaluable_users_returns_nan():
    test_data = {"u1": {"i1": 0.0}}
    recommendations = {"u1": [("i2", 0.9), ("i3", 0.8)]}

    result = compute_accelerated_metrics(
        recommendations=recommendations,
        test_data=test_data,
        cutoff=2,
        relevance_threshold=1.0,
        metric_names=["Precision", "nDCG"],
        device="cpu",
        return_user_metrics=True,
    )

    assert math.isnan(result.results["Precision"])
    assert math.isnan(result.results["nDCG"])
    assert result.user_results["Precision"] == {}
    assert result.user_results["nDCG"] == {}


def test_accelerated_metrics_rejects_unsupported_metric():
    with pytest.raises(ValueError):
        compute_accelerated_metrics(
            recommendations={"u1": [("i1", 1.0)]},
            test_data={"u1": {"i1": 1.0}},
            cutoff=1,
            relevance_threshold=1.0,
            metric_names=["AUC"],
            device="cpu",
        )
