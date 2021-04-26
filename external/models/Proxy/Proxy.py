import operator
import ntpath
import numpy as np
import pandas as pd

from elliot.evaluation.evaluator import Evaluator
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.folder import build_model_folder
from elliot.utils.write import store_recommendation
from elliot.recommender.base_recommender_model import init_charger

np.random.seed(0)


class ProxyRecommender(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a Most Popular recommender.
        :param data: data loader object
        :param path_output_rec_result: path to the directory rec. results
        :param path_output_rec_weight: path to the directory rec. model parameters
        :param args: parameters
        """
        self._random = np.random

        self._params_list = [
            ("_name", "name", "name", "", None, None),
            ("_path", "path", "path", "", None, None)
        ]
        self.autoset_params()
        if not self._name:
            self._name = ntpath.basename(self._path).split(".", 1)[0].split("_", 1)[0]

    @property
    def name(self):
        return self._name

    def train(self):
        print("Reading recommendations")
        self._recommendations = self.read_recommendations(self._path)

        print("Evaluating recommendations")
        self.evaluate()

    def get_recommendations(self, top_k):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(top_k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k):

        nonzero = mask.nonzero()
        num_mask_samples = len(nonzero[0])
        # pairs = list(zip(*nonzero))
        # pairs = list(map(lambda x: (self._data.private_users[x[0]], self._data.private_items[x[1]]), zip(*nonzero)))
        candidate_items = {}
        [candidate_items.setdefault(self._data.private_users[user], set()).add(self._data.private_items[item]) for user, item in zip(*nonzero)]
        recs = {}
        for u, user_recs in self._recommendations.items():
            user_cleaned_recs = []
            user_candidate_items = candidate_items[u]
            for item, prediction in user_recs:
                if item in user_candidate_items:
                    user_cleaned_recs.append((item, prediction))
            recs[u] = user_cleaned_recs
        return recs

    def read_recommendations(self, path):
        recs = {}
        user_recs_items = {}
        column_names = ["userId", "itemId", "prediction", "timestamp"]
        data = pd.read_csv(path, sep="\t", header=None, names=column_names)
        # data = pd.DataFrame(data, columns=column_names)
        user_groups = data.groupby(['userId'])
        for name, group in user_groups:
            recs[name] = sorted(data.loc[group.index][['itemId', 'prediction']].apply(tuple, axis=1).to_list(), key=lambda x: x[1], reverse=True)
            # user_recs_items[name] = data.loc[group.index][['itemId']].to_set()
        return recs



