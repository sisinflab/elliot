import typing as t
from ast import literal_eval
import os
import numpy as np
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class VisualAttribute(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.visual_feature_folder_path = getattr(ns, "visual_features", None)
        self.visual_pca_feature_folder_path = getattr(ns, "visual_pca_features", None)
        self.visual_feat_map_feature_folder_path = getattr(ns, "visual_feat_map_features", None)
        self.images_folder_path = getattr(ns, "images_src_folder", None)

        self.item_mapping = {}
        self.visual_features_shape = None
        self.visual_pca_features_shape = None
        self.visual_feat_map_features_shape = None
        self.image_size_tuple = getattr(ns, "output_image_size", None)
        if self.image_size_tuple:
            self.image_size_tuple = literal_eval(self.image_size_tuple)

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
        ns.__name__ = "VisualAttributes"
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

    def check_items_in_folder(self) -> t.Set[int]:
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
