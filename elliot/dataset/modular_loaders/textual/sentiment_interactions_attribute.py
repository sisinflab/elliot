import typing as t
import os
import numpy as np
import torch
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class SentimentInteractionsTextualAttributes(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.interactions_feature_folder_path = getattr(ns, "interactions_features", None)

        self.item_mapping = {}
        self.user_mapping = {}
        self.interactions_features_shape = None

        inner_items = self.check_interactions_in_folder()

        self.users = users
        self.items = items & inner_items

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = "SentimentInteractionsTextualAttributes"
        ns.object = self
        ns.textual_feature_folder_path = self.interactions_feature_folder_path

        ns.user_mapping = self.user_mapping
        ns.item_mapping = self.item_mapping

        ns.interactions_features_shape = self.interactions_features_shape

        return ns

    def check_interactions_in_folder(self) -> (t.Set[int]):
        items = set()
        if self.interactions_feature_folder_path:
            items_from_folder = [[int(os.path.splitext(filename)[0].split('_')[0]),
                                  int(os.path.splitext(filename)[0].split('_')[1])] for filename in
                                 os.listdir(self.interactions_feature_folder_path)]
            items_from_folder = [item for sublist in items_from_folder for item in sublist]
            items = items.union(items_from_folder)
            self.interactions_features_shape = np.load(os.path.join(self.interactions_feature_folder_path,
                                                                    os.listdir(self.interactions_feature_folder_path)[
                                                                        0])).shape[-1]
        if items:
            self.item_mapping = {item: val for val, item in enumerate(items)}

        return items

    def get_all_features(self, public_items):
        def add_zero(s):
            zeros_to_add = num_digits - len(s)
            return ''.join(['0' for _ in range(zeros_to_add)]) + s
        num_digits = len(str(int(list(public_items.keys())[-1])))
        all_interactions_private = [
            torch.from_numpy(np.load(os.path.join(self.interactions_feature_folder_path, inter))) for inter in
            os.listdir(self.interactions_feature_folder_path)]
        all_interactions_public = ['_'.join(
            [add_zero(str(public_items[float(inter[:-4].split('_')[0])])),
             add_zero(str(public_items[float(inter[:-4].split('_')[1])]))]) for inter in
            os.listdir(self.interactions_feature_folder_path)]
        sorted_indices_public = sorted(range(len(all_interactions_public)), key=lambda k: all_interactions_public[k])
        all_interactions_private = [all_interactions_private[index] for index in sorted_indices_public]
        return all_interactions_private
