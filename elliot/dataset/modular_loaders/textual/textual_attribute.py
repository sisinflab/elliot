import typing as t
import os
import numpy as np
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class TextualAttribute(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.textual_feature_folder_path = getattr(ns, "textual_features", None)

        self.item_mapping = {}
        self.textual_features_shape = None

        inner_items = self.check_items_in_folder()

        self.users = users
        self.items = items & inner_items

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = "TextualAttribute"
        ns.object = self
        ns.textual_feature_folder_path = self.textual_feature_folder_path

        ns.item_mapping = self.item_mapping

        ns.textual_features_shape = self.textual_features_shape

        return ns

    def check_items_in_folder(self) -> t.Set[int]:
        items = set()
        if self.textual_feature_folder_path:
            items_folder = os.listdir(self.textual_feature_folder_path)
            items = items.union(set([int(f.split('.')[0]) for f in items_folder]))
            self.textual_features_shape = np.load(os.path.join(self.textual_feature_folder_path,
                                                               items_folder[0])).shape[0]
        if items:
            self.item_mapping = {item: val for val, item in enumerate(items)}
        return items

    def get_all_features(self):
        return self.get_all_textual_features()

    def get_all_textual_features(self):
        all_features = np.empty((len(self.items), self.textual_features_shape))
        if self.textual_feature_folder_path:
            for key, value in self.item_mapping.items():
                all_features[value] = np.load(self.textual_feature_folder_path + '/' + str(key) + '.npy')
        return all_features
