"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import os

import pandas as pd
from datetime import datetime
import json
from enum import Enum

from elliot.evaluation.statistical_significance import PairedTTest, WilcoxonTest

_eval_results = "test_results"
_eval_statistical_results = "test_statistical_results"


class StatTest(Enum):
    PairedTTest = [PairedTTest, "paired_ttest"]
    WilcoxonTest = [WilcoxonTest, "wilcoxon_test"]


class ResultHandler:
    def __init__(self, rel_threshold=1):
        self.oneshot_recommenders = {}
        self.ks = list()
        self.rel_threshold = rel_threshold

    def add_oneshot_recommender(self, **kwargs):
        new_ks = set(kwargs["test_results"].keys())
        [self.ks.append(k) for k in new_ks if k not in self.ks]
        # self.oneshot_recommenders[kwargs["name"].split("_")[0]] = [kwargs]
        self.oneshot_recommenders[kwargs["name"]] = [kwargs]

    def save_best_results(self, output=''):
        global_results = dict(self.oneshot_recommenders)
        for k in self.ks:
            results = {}
            for rec in global_results.keys():
                for result in global_results[rec]:
                    results.update({result['params']['name']: result[_eval_results][k]})
            info = pd.DataFrame.from_dict(results, orient='index')
            info.insert(0, 'model', info.index)
            info.to_csv(os.path.abspath(os.sep.join([output,
                f'rec_cutoff_{k}_relthreshold_{self.rel_threshold}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.tsv'])),
                sep='\t', index=False)

    def save_best_results_as_triplets(self, output='../results/'):
        global_results = dict(self.oneshot_recommenders)
        for k in self.ks:
            results = {}
            for rec in global_results.keys():
                for result in global_results[rec]:
                    results.update({result['params']['name']: result[_eval_results][k]})
            info = pd.DataFrame.from_dict(results, orient='index')
            info.insert(0, 'model', info.index)
            triplets = info.set_index("model").stack().reset_index()
            triplets.to_csv(os.path.abspath(os.sep.join([output,
                f'triplets_rec_cutoff_{k}_relthreshold_{self.rel_threshold}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.tsv'])),
                sep='\t', index=False, header=["model", "metric", "value"])

    def save_best_models(self, output='../results/', default_metric = "nDCG", default_k = [10]):
        global_results = dict(self.oneshot_recommenders)
        k = default_k[0]
        models = [{"default_validation_metric": default_metric,
                   "default_validation_cutoff": k,
                   "rel_threshold": self.rel_threshold}]
        for rec in global_results.keys():
            for model in global_results[rec]:
                models.append({"meta": model["params"]["meta"].__dict__, "recommender": rec,
                               "configuration": {key: value for key, value in model["params"].items() if
                                                 key != 'meta'}})
        with open(os.path.abspath(os.sep.join([output,
                f'bestmodelparams_cutoff_{k}_relthreshold_{self.rel_threshold}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.json'])),
                mode='w') as f:
            json.dump(models, f, indent=4)

    def save_best_statistical_results(self, stat_test, output='../results/'):
        global_results = dict(self.oneshot_recommenders)
        for k in self.ks:
            results = []
            paired_list = []
            for rec_0, rec_0_model in global_results.items():
                for rec_1, rec_1_model in global_results.items():
                    if (rec_0 != rec_1) & ((rec_0, rec_1) not in paired_list):
                        paired_list.append((rec_0, rec_1))
                        paired_list.append((rec_1, rec_0))

                        metrics = rec_0_model[0][_eval_statistical_results][k].keys()

                        # common_users = []
                        for metric_name in metrics:
                            array_0 = rec_0_model[0][_eval_statistical_results][k][metric_name]
                            array_1 = rec_1_model[0][_eval_statistical_results][k][metric_name]

                            common_users = stat_test.value[0].common_users(array_0, array_1)

                            p_value = stat_test.value[0].compare(array_0, array_1, common_users)

                            results.append((rec_0_model[0]['params']['name'],
                                            rec_1_model[0]['params']['name'],
                                            metric_name,
                                            p_value))
                            results.append((rec_1_model[0]['params']['name'],
                                            rec_0_model[0]['params']['name'],
                                            metric_name,
                                            p_value))

            with open(os.path.abspath(os.sep.join([output,
                    f'stat_{stat_test.value[1]}_cutoff_{k}_relthreshold_{self.rel_threshold}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.tsv'])),
                    "w") as f:
                for tup in results:
                    f.write(f"{tup[0]}\t{tup[1]}\t{tup[2]}\t{tup[3]}\n")


class HyperParameterStudy:
    def __init__(self, rel_threshold=1):
        self.trials = {}
        self.ks = set()
        self.rel_threshold = rel_threshold

    def add_trials(self, obj):
        self.ks.update(set(obj.results[0]["test_results"].keys()))
        name = obj.results[0]["params"]["name"].split("_")[0]
        self.trials[name] = obj.results

    def save_trials(self, output='../results/'):
        for k in self.ks:
            for rec, performance in self.trials.items():
                results = {}
                for result in performance:
                    results.update({result['params']['name']: result[_eval_results][k]})
                info = pd.DataFrame.from_dict(results, orient='index')
                info.insert(0, 'model', info.index)
                info.to_csv(os.path.abspath(os.sep.join([output,
                    f'rec_{rec}_cutoff_{k}_relthreshold_{self.rel_threshold}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.tsv'])),
                    sep='\t', index=False)

    def save_trials_as_triplets(self, output='../results/'):
        for k in self.ks:
            for rec, performance in self.trials.items():
                results = {}
                for result in performance:
                    results.update({result['params']['name']: result[_eval_results][k]})
                info = pd.DataFrame.from_dict(results, orient='index')
                info.insert(0, 'model', info.index)
                triplets = info.set_index("model").stack().reset_index()
                triplets.to_csv(os.path.abspath(os.sep.join([output,
                    f'triplets_rec_{rec}_cutoff_{k}_relthreshold_{self.rel_threshold}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.tsv'])),
                    sep='\t', index=False, header=["model", "metric", "value"])
