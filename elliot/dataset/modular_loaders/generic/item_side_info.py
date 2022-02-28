import typing as t
import os
import numpy as np
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class ItemSideAttribute(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.side_feature_path = getattr(ns, "side_features", None)
        self.side_feature_shape = None

        self.item_mapping = {}

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
        ns.__name__ = "ItemSideAttributes"
        ns.object = self
        ns.side_feature_path = self.side_feature_path

        ns.item_mapping = self.item_mapping

        return ns

    def check_items_in_folder(self) -> t.Set[int]:
        items = set()
        if self.side_feature_path:
            items_folder = os.listdir(self.side_feature_path)
            items = items.union(set([int(f.split('.')[0]) for f in items_folder]))
            self.side_feature_shape = np.load(self.side_feature_path + items_folder[0]).shape[-1]
        if items:
            self.item_mapping = {item: val for val, item in enumerate(items)}
        return items

    def get_all_features(self):
        all_features = np.empty((len(self.items), self.side_feature_shape))
        for i, file in enumerate(list(self.items)):
            all_features[i] = np.load(self.side_feature_path + '/' + str(file) + '.npy')
        return all_features
