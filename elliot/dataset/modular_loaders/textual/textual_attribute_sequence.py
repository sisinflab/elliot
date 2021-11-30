import pickle
import typing as t
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class TextualAttributeSequence(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.textual_feature_file_path = getattr(ns, "textual_features", None)

        self.textual_features = self.check_items_in_file()

        self.users = users
        self.items = items

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = "TextualAttributeSequence"
        ns.object = self
        ns.textual_features = self.textual_features

        return ns

    def check_items_in_file(self):
        if self.textual_feature_file_path:
            textual_features = pickle.load(open(self.textual_feature_file_path, "rb"))
        return textual_features