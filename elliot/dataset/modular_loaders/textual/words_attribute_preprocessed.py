import typing as t
import numpy as np
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class WordsTextualAttributesPreprocessed(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.all_item_texts = getattr(ns, "all_item_texts", None)
        self.all_user_texts = getattr(ns, "all_user_texts", None)
        self.embed_vocabulary = getattr(ns, "embed_vocabulary", None)
        self.item_to_user = getattr(ns, "item_to_user", None)
        self.item_to_user_to_item = getattr(ns, "item_to_user_to_item", None)
        self.user_to_item = getattr(ns, "user_to_item", None)
        self.user_to_item_to_user = getattr(ns, "user_to_item_to_user", None)

        self.all_item_texts_features = None
        self.all_user_texts_features = None
        self.embed_vocabulary_features = None
        self.item_to_user_features = None
        self.item_to_user_to_item_features = None
        self.user_to_item_features = None
        self.user_to_item_to_user_features = None

        self.all_item_texts_shape = None
        self.all_user_texts_shape = None
        self.embed_vocabulary_shape = None
        self.item_to_user_shape = None
        self.item_to_user_to_item_shape = None
        self.user_to_item_shape = None
        self.user_to_item_to_user_shape = None

        self.users = users
        self.items = items

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = "WordsTextualAttributesPreprocessed"
        ns.object = self
        return ns

    def load_all_features(self):
        self.all_item_texts_features = np.load(self.all_item_texts)
        self.all_item_texts_shape = self.all_item_texts_features.shape
        self.all_user_texts_features = np.load(self.all_user_texts)
        self.all_user_texts_shape = self.all_user_texts_features.shape
        self.embed_vocabulary_features = np.load(self.embed_vocabulary)
        self.embed_vocabulary_shape = self.embed_vocabulary_features.shape
        self.item_to_user_features = np.load(self.item_to_user)
        self.item_to_user_shape = self.item_to_user_features.shape
        self.user_to_item_features = np.load(self.user_to_item)
        self.user_to_item_shape = self.user_to_item_features.shape
        self.item_to_user_to_item_features = np.load(self.item_to_user_to_item)
        self.item_to_user_to_item_shape = self.item_to_user_to_item_features.shape
        self.user_to_item_to_user_features = np.load(self.user_to_item_to_user)
        self.user_to_item_to_user_shape = self.user_to_item_to_user_features.shape
