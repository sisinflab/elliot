"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pandas as pd
from datetime import datetime

from evaluation.statistical_significance import PairedTTest

_eval_results = "test_results"
_eval_statistical_results = "test_statistical_results"

class ResultHandler:
    def __init__(self):
        self.multishot_recommenders = {}
        self.oneshot_recommenders = {}
        self.trials = {}

    def add_multishot_recommender(self, obj):
        name = obj.results[0]["params"]["name"].split("_")[0]
        self.multishot_recommenders[name] = obj.results

    def add_trials(self, obj):
        name = obj.results[0]["params"]["name"].split("_")[0]
        self.trials[name] = obj.results

    def add_oneshot_recommender(self, **kwargs):
        # rec = {}
        # rec["name"] = kwargs["name"]
        # rec["loss"] = kwargs["loss"]
        # rec["params"] = kwargs["params"]
        # rec["results"] = kwargs["results"]
        # rec["statistical_results"] = kwargs["statistical_results"]
        self.oneshot_recommenders[kwargs["name"].split("_")[0]] = [kwargs]

    def get_best_result(self):
        bests = {}
        for recommender in self.multishot_recommenders.keys():
            min_val = np.argmin([i["loss"] for i in self.multishot_recommenders[recommender]])
            best_model_loss = self.multishot_recommenders[recommender][min_val]["loss"]
            best_model_params = self.multishot_recommenders[recommender][min_val]["params"]
            best_model_results = self.multishot_recommenders[recommender][min_val]["results"]
            best_model_statistical_results = self.multishot_recommenders[recommender][min_val]["statistical_results"]
            best_test_model_results = self.multishot_recommenders[recommender][min_val]["test_results"]
            best_test_model_statistical_results = self.multishot_recommenders[recommender][min_val]["test_statistical_results"]
            bests[recommender] = [{"loss": best_model_loss,
                                   "params": best_model_params,
                                   "results": best_model_results,
                                   "statistical_results": best_model_statistical_results,
                                   "test_results": best_test_model_results,
                                   "test_statistical_results": best_test_model_statistical_results}]
        return bests

    def save_results(self, output='../results/', best=False):
        global_results = dict(self.oneshot_recommenders,
                              **self.get_best_result() if best else self.multishot_recommenders)
        for rec in global_results.keys():
            results = {}
            for result in global_results[rec]:
                results.update({result['params']['name']: result[_eval_results]})
            info = pd.DataFrame.from_dict(results, orient='index')
            info.insert(0, 'model', info.index)
            info.to_csv(f'{output}rec_{rec}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.tsv', sep='\t', index=False)

    def save_trials(self, output='../results/'):
        for rec, performance in self.trials.items():
            results = {}
            for result in performance:
                results.update({result['params']['name']: result[_eval_results]})
            info = pd.DataFrame.from_dict(results, orient='index')
            info.insert(0, 'model', info.index)
            info.to_csv(f'{output}rec_{rec}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.tsv', sep='\t', index=False)

    def save_best_results(self, output='../results/'):
        global_results = dict(self.oneshot_recommenders, **self.get_best_result())
        results = {}
        for rec in global_results.keys():
            for result in global_results[rec]:
                results.update({result['params']['name']: result[_eval_results]})
        info = pd.DataFrame.from_dict(results, orient='index')
        info.insert(0, 'model', info.index)
        info.to_csv(f'{output}rec_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.tsv', sep='\t', index=False)

    def save_best_statistical_results(self, output='../results/'):
        global_results = dict(self.oneshot_recommenders, **self.get_best_result())
        results = []
        paired_list = []
        for rec_0, rec_0_model in global_results.items():
            for rec_1, rec_1_model in global_results.items():
                if (rec_0 != rec_1) & ((rec_0, rec_1) not in paired_list):
                    paired_list.append((rec_0, rec_1))
                    paired_list.append((rec_1, rec_0))

                    metrics = rec_0_model[0][_eval_statistical_results].keys()

                    common_users = []
                    for metric_name in metrics:
                        array_0 = rec_0_model[0][_eval_statistical_results][metric_name]
                        array_1 = rec_1_model[0][_eval_statistical_results][metric_name]

                        if not common_users:
                            common_users = PairedTTest.common_users(array_0, array_1)

                        p_value = PairedTTest.compare(array_0, array_1, common_users)

                        results.append((rec_0_model[0]['params']['name'],
                                        rec_1_model[0]['params']['name'],
                                        metric_name,
                                        p_value))
                        results.append((rec_1_model[0]['params']['name'],
                                        rec_0_model[0]['params']['name'],
                                        metric_name,
                                        p_value))

        with open(f'{output}stat_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.tsv', "w") as f:
            for tup in results:
                f.write(f"{tup[0]}\t{tup[1]}\t{tup[2]}\t{tup[3]}\n")

