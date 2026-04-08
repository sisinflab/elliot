from typing import Tuple, Set
from types import SimpleNamespace
from ast import literal_eval
import os
import numpy as np

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
from elliot.utils.enums import AlignmentMode, Materialization
from elliot.utils.registry import side_info_registry


@side_info_registry.register(
    provides="item_features",
    format="dense",
    alignment=AlignmentMode.PAD,
    materialization=Materialization.MMAP
)
class VisualAttribute(AbstractLoader):
    visual_feature_folder_path: str = None
    visual_pca_feature_folder_path: str = None
    visual_feat_map_feature_folder_path: str = None
    images_folder_path: str = None
    image_size_tuple: str = None

    def __init__(self, **params):
        super().__init__(**params)

        self.item_mapping = {}

        self.visual_features_shape = None
        self.visual_pca_features_shape = None
        self.visual_feat_map_features_shape = None

        if self.image_size_tuple:
            self.image_size_tuple = literal_eval(self.image_size_tuple)

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
        ns.visual_feature_folder_path = self.visual_feature_folder_path
        ns.visual_pca_feature_folder_path = self.visual_pca_feature_folder_path
        ns.visual_feat_map_feature_folder_path = self.visual_feat_map_feature_folder_path
        ns.images_folder_path = self.images_folder_path

        ns.item_mapping = self.item_mapping

        ns.visual_features_shape = self.visual_features_shape
        ns.visual_pca_features_shape = self.visual_pca_features_shape
        ns.visual_feat_map_features_shape = self.visual_feat_map_features_shape
        ns.image_size_tuple = self.image_size_tuple

        return ns

    def check_items_in_folder(self) -> Set[int]:
        items = set()
        if self.visual_feature_folder_path:
            items_folder = os.listdir(self.visual_feature_folder_path)
            items = items.union(set([int(f.split('.')[0]) for f in items_folder]))
            self.visual_features_shape = np.load(os.path.join(self.visual_feature_folder_path,
                                                              items_folder[0])).shape[0]
        if self.visual_pca_feature_folder_path:
            items_folder = os.listdir(self.visual_feature_folder_path)
            items = items.union(set([int(f.split('.')[0]) for f in items_folder]))
            self.visual_pca_features_shape = np.load(os.path.join(self.visual_pca_feature_folder_path,
                                                                  items_folder[0])).shape[0]
        if self.visual_feat_map_feature_folder_path:
            items_folder = os.listdir(self.visual_feature_folder_path)
            items = items.union(set([int(f.split('.')[0]) for f in items_folder]))
            self.visual_feat_map_features_shape = np.load(os.path.join(self.visual_feat_map_feature_folder_path,
                                                          items_folder[0])).shape
        if self.images_folder_path:
            items_folder = os.listdir(self.visual_feature_folder_path)
            items = items.union(set([int(f.split('.')[0]) for f in items_folder]))

        if items:
            self.item_mapping = {item: val for val, item in enumerate(items)}
        return items

    def get_all_features(self):
        return self.get_all_visual_features()

    def get_all_visual_features(self):
        all_features = np.empty((len(self.items), self.visual_features_shape))
        if self.visual_feature_folder_path:
            for key, value in self.item_mapping.items():
                all_features[value] = np.load(self.visual_feature_folder_path + '/' + str(key) + '.npy')
        return all_features

    def get_all_visual_pca_features(self):
        all_features = np.empty((len(self.items), self.visual_pca_features_shape))
        if self.visual_pca_feature_folder_path:
            for key, value in self.item_mapping.items():
                all_features[value] = np.load(self.visual_pca_feature_folder_path + '/' + str(key) + '.npy')
        return all_features

    def get_all_visual_feat_map_features(self):
        all_features = np.empty((len(self.items), self.visual_feat_map_features_shape))
        if self.visual_feat_map_feature_folder_path:
            for key, value in self.item_mapping.items():
                all_features[value] = np.load(self.visual_feat_map_feature_folder_path + '/' + str(key) + '.npy')
        return all_features

