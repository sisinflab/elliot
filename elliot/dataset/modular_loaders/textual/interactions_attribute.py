import typing as t
import os
import pandas as pd
import numpy as np
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class InteractionsTextualAttributes(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.interactions_feature_folder_path = getattr(ns, "interactions_features", None)
        self.interactions_path = getattr(ns, "interactions", None)

        self.item_mapping = {}
        self.user_mapping = {}
        self.interactions_features_shape = None

        inner_users, inner_items = self.check_interactions_in_folder()

        self.users = users & inner_users
        self.items = items & inner_items

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = "InteractionsTextualAttributes"
        ns.object = self
        ns.textual_feature_folder_path = self.interactions_feature_folder_path
        ns.interactions_path = self.interactions_path

        ns.user_mapping = self.user_mapping
        ns.item_mapping = self.item_mapping

        ns.interactions_features_shape = self.interactions_features_shape

        return ns

    def check_interactions_in_folder(self) -> (t.Set[int], t.Set[int]):
        users = set()
        items = set()
        if self.interactions_feature_folder_path and self.interactions_path:
            all_interactions = pd.read_csv(self.interactions_path, sep='\t', header=None)
            interactions_folder = os.listdir(self.interactions_feature_folder_path)
            users = users.union(all_interactions[0].unique().tolist())
            items = items.union(all_interactions[1].unique().tolist())
            self.interactions_features_shape = np.load(os.path.join(self.interactions_feature_folder_path,
                                                                    interactions_folder[0])).shape
        if users:
            self.user_mapping = {user: val for val, user in enumerate(users)}
        if items:
            self.item_mapping = {item: val for val, item in enumerate(items)}

        return users, items

    def get_all_features(self):
        user_item_interactions = pd.read_csv(self.interactions_path, sep='\t', header=None)
        user_item_interactions = user_item_interactions.groupby(0).apply(
            lambda x: x.sort_values(by=[1], ascending=True)).reset_index(drop=True)
        interactions = len(user_item_interactions)
        user_item_features = np.empty((interactions, *self.interactions_features_shape))
        for i, row in user_item_interactions.iterrows():
            user_item_features[i] = np.load(self.interactions_feature_folder_path + '/' + str(row[2]) + '.npy')
        return user_item_features
