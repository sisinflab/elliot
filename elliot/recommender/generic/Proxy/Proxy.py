import ntpath
import numpy as np
import pandas as pd
from tqdm import tqdm

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.base_recommender_model import init_charger


class ProxyRecommender(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a Proxy recommender to evaluate already generated recommendations.
        :param name: data loader object
        :param path: path to the directory rec. results
        :param args: parameters
        """
        self._random = np.random

        self._params_list = [
            ("_name", "name", "name", "", None, None),
            ("_path", "path", "path", "", None, None)
        ]
        self.autoset_params()
        if not self._name:
            self._name = ntpath.basename(self._path).rsplit(".",1)[0]

    @property
    def name(self):
        return self._name

    def train(self):
        self.logger.info("Loading recommendations")
        self._recommendations = self.read_recommendations(self._path)

        self.logger.info("Evaluating recommendations")
        self.evaluate()

    def get_recommendations(self, top_k):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(top_k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k):

        # nonzero = mask.nonzero()
        # zero = np.where(mask==False)
        candidate_items = {}
        setItem = set(range(mask.shape[1]))
        for user in tqdm(range(mask.shape[0])):
            itemFalse = set(np.where(mask[user,:]==False)[0].tolist())
            itemTrue = list(setItem.difference(itemFalse))
            candidate_items[self._data.private_users[user]] = [self._data.private_items[item] for item in itemTrue]
            # [candidate_items.setdefault(self._data.private_users[user], list()).add(
                        # self._data.private_items[item]) for item in itemTrue]
            # for item in range(mask.shape[1]):
            #     if item in itemFalse:
            #         continue
            #     else:
            #         candidate_items.setdefault(self._data.private_users[user], set()).add(
            #             self._data.private_items[item])
        # [candidate_items.setdefault(self._data.private_users[user], set()).add(self._data.private_items[item]) for user, item in zip(*nonzero)]
        recs = {}
        for u, user_recs in self._recommendations.items():
            user_cleaned_recs = []
            user_candidate_items = set(candidate_items[u])
            for p, (item, prediction) in enumerate(user_recs):
                if p >= k:
                    break
                if item in user_candidate_items:
                    user_cleaned_recs.append((item, prediction))
            recs[u] = user_cleaned_recs
        return recs

    def read_recommendations(self, path):
        recs = dict()
        column_names = ["userId", "itemId", "prediction"]
        data = pd.read_csv(path, sep="\t", header=None, names=column_names)
        data = data.sort_values(by='prediction', ascending=False)
        user_groups = data.groupby(['userId'])
        # user_groups = user_groups.sort_values(by='prediction', ascending=False)
        # recs = {name: list(group[['itemId', 'prediction']].itertuples(index=False, name=None)) for name, group in tqdm(user_groups)}
        for name, group in tqdm(user_groups):
            #df.sort_values(by=['col1'])
            recs[name] = list(group[['itemId', 'prediction']].itertuples(index=False, name=None))#data.loc[group.index][['itemId', 'prediction']].apply(tuple, axis=1).to_list()
            # recs[name] = sorted(data.loc[group.index][['itemId', 'prediction']].apply(tuple, axis=1).to_list(), key=lambda x: x[1], reverse=True)
        return recs