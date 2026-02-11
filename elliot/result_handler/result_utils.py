import numpy as np


def _aggregate(fold_results, key, reducer):
    sample = fold_results[0].get("test_results", {})
    if not sample:
        return {}

    cutoffs = list(sample.keys())
    metrics = list(next(iter(sample.values())).keys())
    agg = {}

    for k in cutoffs:
        agg[k] = {}
        for metric in metrics:
            values = [
                fold[key][k][metric]
                for fold in fold_results if k in fold[key]
            ]
            if not values:
                continue
            agg[k][metric] = float(reducer(values))

    return agg


def aggregate_val_folds_results(fold_results, include_test=True):
    if not fold_results:
        return {}

    first, last = fold_results[0], fold_results[-1]

    result = {}

    result["val_results"] = _aggregate(fold_results, "val_results", np.average)
    if include_test:
        result["test_results"] = _aggregate(fold_results, "test_results", np.average)

    result["name"] = first["name"]
    result["params"] = first["params"]

    result["val_statistical_results"] = last["val_statistical_results"]
    if include_test:
        result["test_statistical_results"] = last["test_statistical_results"]

    result["time"] = [r["time"] for r in fold_results]

    return result


def attach_test_fold_stats(best_eval, fold_results):
    if len(fold_results) < 2:
        return
    best_eval["test_mean_results"] = _aggregate(fold_results, "test_results", np.mean)
    best_eval["test_std_results"] = _aggregate(fold_results, "test_results", np.std)
