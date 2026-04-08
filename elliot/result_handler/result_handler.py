"""
Simple results and trials storage for experiments.
"""

import json
import numpy as np
from enum import Enum

from elliot.evaluation.statistical_significance import PairedTTest, WilcoxonTest
from elliot.namespace import ExperimentConfig, ResultsConfig
from elliot.utils import split_metric
from elliot.utils.enums import StatTest
from elliot.utils.write import Writer

_EVAL_RESULTS = "test_results"
_EVAL_STD_RESULTS = "test_std_results"
_EVAL_MEAN_RESULTS = "test_mean_results"
_EVAL_STAT_RESULTS = "test_statistical_results"
_EVAL_TIME = "time"

STAT_TESTS = {
    StatTest.PAIRED_TTEST: PairedTTest,
    StatTest.WILCOXON_TEST: WilcoxonTest
}


class ResultHandler:
    results_config: ResultsConfig

    def __init__(self, config: ExperimentConfig):
        self.writer = Writer()

        self.results_config = config.results
        self.output_folder = config.path_output_rec_performance
        self.paired_ttest = config.evaluation.paired_ttest
        self.wilcoxon_test = config.evaluation.wilcoxon_test
        self.rel_threshold = config.evaluation.relevance_threshold

        self.results = {}  # model_name -> result dict
        self.trials = {}   # model_name -> list[dict]

        self._suffix = f"_relthreshold_{self.rel_threshold}"

    def add_oneshot_recommender(self, **kwargs):
        self.results[kwargs["name"]] = kwargs

    def add_trials(self, obj, name=None):
        results = obj.results if hasattr(obj, "results") else obj
        if not results:
            return
        if name is None:
            name = results[0]["params"]["name"]#.split("_")[0]
        self.trials[name] = results

    def save_outputs(self):
        output = self.output_folder

        if self.results_config.save_performance:
            self.save_results(output=output, triplets=False)
        if self.results_config.save_performance_triplets:
            self.save_results(output=output, triplets=True)

        if self.results_config.save_fold_stats:
            self.save_results(output=output, key="test_mean_results", triplets=False)
            self.save_results(output=output, key="test_std_results", triplets=False)
            if self.results_config.save_fold_stats_triplets:
                self.save_results(output=output, key="test_mean_results", triplets=True)
                self.save_results(output=output, key="test_std_results", triplets=True)

        if self.results_config.save_times:
            self.save_times(output=output)

        if self.results_config.save_best_models:
            metric, cutoff = split_metric(self.results_config.default_metric)
            self.save_best_models(output=output, default_metric=metric, default_k=cutoff)

        if self.results_config.save_trials:
            self.save_trials(output=output)

        if self.results_config.save_statistical:
            if self.paired_ttest:
                self.save_statistical_results(StatTest.PAIRED_TTEST, output=output)
            if self.wilcoxon_test:
                self.save_statistical_results(StatTest.WILCOXON_TEST, output=output)

    # def _is_new_recommender(self, params):
    #     if isinstance(params, SimpleNamespace):
    #         return False
    #     if isinstance(params, dict):
    #         return "meta" not in params
    #     return False
    #
    # def _cutoffs(self, items, key):
    #     for entry in items:
    #         if key in entry:
    #             return list(entry[key].keys())
    #     return []

    # def _collect_cutoff(self, items, key, k):
    #     data = {}
    #     for entry in items:
    #         if key in entry and k in entry[key]:
    #             data[entry["params"]["name"]] = entry[key][k]
    #     return data

    # def _write_table(self, data, output, filename):
    #     if not data:
    #         return
    #     info = pd.DataFrame.from_dict(data, orient="index")
    #     info.insert(0, "model", info.index)
    #     info.to_csv(os.path.abspath(os.sep.join([output, filename])), sep="\t", index=False)
    #
    # def _write_triplets(self, data, output, filename):
    #     if not data:
    #         return
    #     info = pd.DataFrame.from_dict(data, orient="index")
    #     info.insert(0, "model", info.index)
    #     triplets = info.set_index("model").stack().reset_index()
    #     triplets.to_csv(
    #         os.path.abspath(os.sep.join([output, filename])),
    #         sep="\t",
    #         index=False,
    #         header=["model", "metric", "value"],
    #     )

    def save_results(self, output="", key=_EVAL_RESULTS, triplets=False):
        results_dict = {
            name: entry.get(key, {})
            for name, entry in self.results.items()
        }

        values = next(iter(results_dict.values()))

        results = {}
        for k in values.keys():
            results[k] = {name: entry[k] for name, entry in results_dict.items()}

        self.writer.write_results(
            results=results,
            save_folder=output,
            file_name=f"_{key}{self._suffix}",
            header=self.results_config.writer.header,
            ext=self.results_config.writer.ext,
            sep=self.results_config.writer.sep,
            triplets=triplets
        )

    def save_times(self, output=""):
        data = {
            name: {_EVAL_TIME: entry[_EVAL_TIME]}
            for name, entry in self.results.items() if _EVAL_TIME in entry
        }

        self.writer.write_times(
            data=data,
            save_folder=output,
            file_name=self._suffix,
            header=self.results_config.writer.header,
            ext=self.results_config.writer.ext,
            sep=self.results_config.writer.sep
        )

    def save_trials(self, output=""):
        if not self.trials:
            return

        trials = self._normalize(self.trials)

        if "json" in self.results_config.trials_formats:
            self.writer.write_trials(
                trials=trials,
                save_folder=output,
                file_name=self._suffix,
                frmt="json"
            )

        if "tabular" in self.results_config.trials_formats:
            trials_dict = {}
            for model_name, trials_list in trials.items():
                rows = []
                for idx, entry in enumerate(trials_list):
                    rows.append({
                        "model": model_name,
                        "trial": idx,
                        "loss": entry.get("loss"),
                        "status": entry.get("status"),
                        "val_metric": entry.get("val_metric"),
                        "time": self._to_json(entry.get("time")),
                        "objective": self._to_json(entry.get("objective")),
                        "params": self._to_json(entry.get("params")),
                        "val_results": self._to_json(entry.get("val_results"))
                    })
                trials_dict[model_name] = rows

            self.writer.write_trials(
                trials=trials_dict,
                save_folder=output,
                file_name=self._suffix,
                frmt="tabular",
                header=self.results_config.writer.header,
                ext=self.results_config.writer.ext,
                sep=self.results_config.writer.sep
            )

    def save_best_models(self, output="../results/", default_metric="nDCG", default_k=10):
        models = [{
            "default_validation_metric": default_metric,
            "default_validation_cutoff": default_k,
            "rel_threshold": self.rel_threshold,
        }]

        results = {name: entry.get("params", {}) for name, entry in self.results.items()}
        results = self._normalize(results)

        for model_name, params in results.items():
            models += [{
                "meta": params["meta"],
                "recommender": model_name,
                "configuration": {key: value for key, value in params.items() if key != "meta"},
            }]

        self.writer.write_params(
            params=models,
            save_folder=output,
            file_name=self._suffix,
        )

    def save_statistical_results(self, stat_test, output="../results/"):
        results_dict = {
            name: entry.get(_EVAL_STAT_RESULTS, {})
            for name, entry in self.results.items()
        }

        values = next(iter(results_dict.values()))

        results = {}
        for k in values.keys():
            results_list = []
            paired = set()
            for i, left in results_dict.items():
                for j, right in results_dict.items():
                    if i == j or (j, i) in paired:
                        continue
                    paired.add((i, j))
                    metrics = left[k].keys()
                    for metric_name in metrics:
                        array_0 = left[k][metric_name]
                        array_1 = right[k][metric_name]
                        common_users = STAT_TESTS[stat_test].common_users(array_0, array_1)
                        p_value = STAT_TESTS[stat_test].compare(array_0, array_1, common_users)
                        results_list.append((i, j, metric_name, p_value))
                        results_list.append((j, i, metric_name, p_value))
            results[k] = results_list

        self.writer.write_statistical_results(
            results=results,
            save_folder=output,
            file_name=self._suffix,
            header=self.results_config.writer.header,
            ext=self.results_config.writer.ext,
            sep=self.results_config.writer.sep,
            stat_test=stat_test.value
        )

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

    def _normalize(self, value):
        # if isinstance(value, SimpleNamespace):
        #     return {k: self._normalize(v) for k, v in vars(value).items()}
        if isinstance(value, dict):
            return {k: self._normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._normalize(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Enum):
            return value.value
        return value

    def _to_json(self, value):
        if value is None:
            return None
        return json.dumps(self._normalize(value), ensure_ascii=True)
