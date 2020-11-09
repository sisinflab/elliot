import numpy as np
import pandas as pd
from datetime import datetime

class ResultHandler:
    def __init__(self):
        self.multishot_recommenders = {}
        self.oneshot_recommenders = {}

    def add_multishot_recommender(self, obj):
        name = obj.results[0]["params"]["name"].split("_")[0]
        self.multishot_recommenders[name] = obj.results

    def add_oneshot_recommender(self, name, loss, params, results):
        rec = {}
        rec["name"] = name
        rec["loss"] = loss
        rec["params"] = params
        rec["results"] = results
        self.oneshot_recommenders[name.split("_")[0]] = [rec]

    def get_best_result(self):
        bests = {}
        for recommender in self.multishot_recommenders.keys():
            min_val = np.argmin([i["loss"] for i in self.multishot_recommenders[recommender]])
            best_model_loss = self.multishot_recommenders[recommender][min_val]["loss"]
            best_model_params = self.multishot_recommenders[recommender][min_val]["params"]
            best_model_results = self.multishot_recommenders[recommender][min_val]["results"]
            bests[recommender] = {"loss": best_model_loss, "params": best_model_params, "results": best_model_results}
        return bests

    def save_results(self, output='../results/', best=False):
        global_results = dict(self.oneshot_recommenders, **self.get_best_result() if best else self.multishot_recommenders)
        for rec in global_results.keys():
            results = {}
            for result in global_results[rec]:
                results.update({result['params']['name']: result['results']})
            info = pd.DataFrame.from_dict(results, orient='index')
            info.insert(0, 'model', info.index)
            info.reset_index()
            info.to_csv(f'{output}rec_{rec}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.tsv', sep='\t', index=False)