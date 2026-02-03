"""
Simple results and trials storage for experiments.
"""

import json
import os
from datetime import datetime
from enum import Enum
from types import SimpleNamespace

import pandas as pd
import numpy as np

from elliot.evaluation.statistical_significance import PairedTTest, WilcoxonTest

_EVAL_RESULTS = "test_results"
_EVAL_STD_RESULTS = "test_std_results"
_EVAL_MEAN_RESULTS = "test_mean_results"
_EVAL_STAT_RESULTS = "test_statistical_results"
_EVAL_TIME = "time"


class StatTest(Enum):
    PairedTTest = [PairedTTest, "paired_ttest"]
    WilcoxonTest = [WilcoxonTest, "wilcoxon_test"]


def _timestamp():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


class ResultHandler:
    def __init__(self, rel_threshold=1):
        self.rel_threshold = rel_threshold
        self.results = {}  # model_name -> result dict
        self.trials = {}   # model_name -> list[dict]

    def add_oneshot_recommender(self, **kwargs):
        self.results[kwargs["params"]["name"]] = kwargs

    def add_trials(self, obj, name=None):
        results = obj.results if hasattr(obj, "results") else obj
        if not results:
            return
        if name is None:
            name = results[0]["params"]["name"].split("_")[0]
        self.trials[name] = results

    def _normalize(self, value):
        if isinstance(value, SimpleNamespace):
            return {k: self._normalize(v) for k, v in vars(value).items()}
        if isinstance(value, dict):
            return {k: self._normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._normalize(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _to_json(self, value):
        if value is None:
            return None
        return json.dumps(self._normalize(value), ensure_ascii=True)

    def _is_new_recommender(self, params):
        if isinstance(params, SimpleNamespace):
            return False
        if isinstance(params, dict):
            return "meta" not in params
        return False

    def _cutoffs(self, items, key):
        for entry in items:
            if key in entry:
                return list(entry[key].keys())
        return []

    def _collect_cutoff(self, items, key, k):
        data = {}
        for entry in items:
            if key in entry and k in entry[key]:
                data[entry["params"]["name"]] = entry[key][k]
        return data

    def _write_table(self, data, output, filename):
        if not data:
            return
        info = pd.DataFrame.from_dict(data, orient="index")
        info.insert(0, "model", info.index)
        info.to_csv(os.path.abspath(os.sep.join([output, filename])), sep="\t", index=False)

    def _write_triplets(self, data, output, filename):
        if not data:
            return
        info = pd.DataFrame.from_dict(data, orient="index")
        info.insert(0, "model", info.index)
        triplets = info.set_index("model").stack().reset_index()
        triplets.to_csv(
            os.path.abspath(os.sep.join([output, filename])),
            sep="\t",
            index=False,
            header=["model", "metric", "value"],
        )

    def save_results(self, output="", key=_EVAL_RESULTS, triplets=False):
        items = list(self.results.values())
        for k in self._cutoffs(items, key):
            data = self._collect_cutoff(items, key, k)
            prefix = "triplets_rec" if triplets else "rec"
            name = f"{prefix}_cutoff_{k}_relthreshold_{self.rel_threshold}_{_timestamp()}.tsv"
            if triplets:
                self._write_triplets(data, output, name)
            else:
                self._write_table(data, output, name)

    def save_times(self, output=""):
        data = {entry["params"]["name"]: entry.get(_EVAL_TIME) for entry in self.results.values() if _EVAL_TIME in entry}
        self._write_table(
            data,
            output,
            f"rec_training_time_relthreshold_{self.rel_threshold}_{_timestamp()}.tsv",
        )

    def save_trials(self, output="", formats=("json", "tsv")):
        if not self.trials:
            return
        if isinstance(formats, str):
            formats = [formats]
        for model_name, trials in self.trials.items():
            if not trials:
                continue
            first_params = trials[0].get("params", {}) if isinstance(trials[0], dict) else {}
            if isinstance(first_params, SimpleNamespace):
                first_params = vars(first_params)
            if not self._is_new_recommender(first_params):
                continue
            normalized = [self._normalize(entry) for entry in trials]
            if "json" in formats:
                filename = f"trials_{model_name}_relthreshold_{self.rel_threshold}_{_timestamp()}.json"
                with open(os.path.abspath(os.sep.join([output, filename])), "w", encoding="utf-8") as handle:
                    json.dump(normalized, handle, indent=2, ensure_ascii=True)
            if "tsv" in formats:
                rows = []
                for idx, entry in enumerate(normalized):
                    rows.append({
                        "model": model_name,
                        "trial": idx,
                        "loss": entry.get("loss"),
                        "status": entry.get("status"),
                        "val_metric": entry.get("val_metric"),
                        "test_metric": entry.get("test_metric"),
                        "time": self._to_json(entry.get("time")),
                        "objective": self._to_json(entry.get("objective")),
                        "params": self._to_json(entry.get("params")),
                        "val_results": self._to_json(entry.get("val_results")),
                        "test_results": self._to_json(entry.get("test_results")),
                    })
                info = pd.DataFrame(rows)
                filename = f"trials_{model_name}_relthreshold_{self.rel_threshold}_{_timestamp()}.tsv"
                info.to_csv(os.path.abspath(os.sep.join([output, filename])), sep="\t", index=False)

    def save_best_models(self, output="../results/", default_metric="nDCG", default_k=10):
        models = [{
            "default_validation_metric": default_metric,
            "default_validation_cutoff": default_k,
            "rel_threshold": self.rel_threshold,
        }]
        for rec, entry in self.results.items():
            params = entry.get("params", {})
            if isinstance(params, SimpleNamespace):
                params = vars(params)
            if not self._is_new_recommender(params):
                continue
            meta_obj = None
            if isinstance(params, dict):
                meta_obj = params.get("meta")
            else:
                meta_obj = getattr(params, "meta", None)
            meta = self._normalize(meta_obj) if meta_obj is not None else {}
            models.append({
                "meta": meta,
                "recommender": rec,
                "configuration": {key: value for key, value in params.items() if key != "meta"},
            })
        with open(os.path.abspath(os.sep.join([output,
                f"bestmodelparams_cutoff_{default_k}_relthreshold_{self.rel_threshold}_{_timestamp()}.json"])),
                mode="w") as f:
            json.dump(models, f, indent=4)

    def save_statistical_results(self, stat_test, output="../results/"):
        items = list(self.results.values())
        for k in self._cutoffs(items, _EVAL_STAT_RESULTS):
            results = []
            paired = set()
            for i, left in enumerate(items):
                for j, right in enumerate(items):
                    if i == j or (j, i) in paired:
                        continue
                    paired.add((i, j))
                    metrics = left[_EVAL_STAT_RESULTS][k].keys()
                    for metric_name in metrics:
                        array_0 = left[_EVAL_STAT_RESULTS][k][metric_name]
                        array_1 = right[_EVAL_STAT_RESULTS][k][metric_name]
                        common_users = stat_test.value[0].common_users(array_0, array_1)
                        p_value = stat_test.value[0].compare(array_0, array_1, common_users)
                        results.append((left["params"]["name"], right["params"]["name"], metric_name, p_value))
                        results.append((right["params"]["name"], left["params"]["name"], metric_name, p_value))

            with open(os.path.abspath(os.sep.join([output,
                    f"stat_{stat_test.value[1]}_cutoff_{k}_relthreshold_{self.rel_threshold}_{_timestamp()}.tsv"])),
                    "w") as f:
                for tup in results:
                    f.write(f"{tup[0]}\t{tup[1]}\t{tup[2]}\t{tup[3]}\n")

    # Backward-compatible wrappers
    def save_best_results(self, output=""):
        self.save_results(output=output, key=_EVAL_RESULTS, triplets=False)

    def save_best_results_std(self, output=""):
        self.save_results(output=output, key=_EVAL_STD_RESULTS, triplets=False)

    def save_best_results_mean(self, output=""):
        self.save_results(output=output, key=_EVAL_MEAN_RESULTS, triplets=False)

    def save_best_results_as_triplets(self, output="../results/"):
        self.save_results(output=output, key=_EVAL_RESULTS, triplets=True)

    def save_best_results_std_as_triplets(self, output="../results/"):
        self.save_results(output=output, key=_EVAL_STD_RESULTS, triplets=True)
