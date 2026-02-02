"""
Simple results and trials storage for experiments.
"""

import json
import os
from datetime import datetime
from enum import Enum

import pandas as pd

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

    def save_best_models(self, output="../results/", default_metric="nDCG", default_k=10):
        models = [{
            "default_validation_metric": default_metric,
            "default_validation_cutoff": default_k,
            "rel_threshold": self.rel_threshold,
        }]
        for rec, entry in self.results.items():
            models.append({
                "meta": entry["params"]["meta"].__dict__,
                "recommender": rec,
                "configuration": {key: value for key, value in entry["params"].items() if key != "meta"},
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

