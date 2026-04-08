from typing import Tuple, Set
from types import SimpleNamespace
import os
import numpy as np

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
from elliot.utils.enums import AlignmentMode, Materialization
from elliot.utils.registry import side_info_registry


@side_info_registry.register(
    provides="item_features",
    format="sparse",
    alignment=AlignmentMode.PAD,
    materialization=Materialization.LAZY,
)
class TextualAttribute(AbstractLoader):
    textual_feature_folder_path: str = None

    def __init__(self, **params):
        super().__init__(**params)

        self.item_mapping = {}
        self.textual_features_shape = None

        inner_items = self.check_items_in_folder()
        self.items = self.items & inner_items

    def get_mapped(self) -> Tuple[Set[int], Set[int]]:
        return self.users, self.items

    def filter(self, users: Set[int], items: Set[int]):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = self.name
        ns.object = self
        ns.textual_feature_folder_path = self.textual_feature_folder_path
        ns.item_mapping = self.item_mapping
        ns.textual_features_shape = self.textual_features_shape
        return ns

    def check_items_in_folder(self) -> Set[int]:
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
