import typing as t
import numpy as np
import json
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class WordsTextualAttributes(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.users_vocabulary_features_path = getattr(ns, "users_vocabulary_features", None)
        self.items_vocabulary_features_path = getattr(ns, "items_vocabulary_features", None)
        self.users_tokens_path = getattr(ns, "users_tokens", None)
        self.items_tokens_path = getattr(ns, "items_tokens", None)
        self.pos_users_path = getattr(ns, "pos_users", None)
        self.pos_items_path = getattr(ns, "pos_items", None)

        self.item_mapping = {}
        self.user_mapping = {}
        self.word_feature_shape = None
        self.users_word_features = None
        self.items_word_features = None
        self.users_tokens = None
        self.items_tokens = None
        self.pos_users = None
        self.pos_items = None

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
        ns.__name__ = "WordsTextualAttributes"
        ns.object = self
        ns.users_vocabulary_features_path = self.users_vocabulary_features_path
        ns.items_vocabulary_featutes_path = self.items_vocabulary_features_path
        ns.users_tokens_path = self.users_tokens_path
        ns.items_tokens_path = self.items_tokens_path

        ns.user_mapping = self.user_mapping
        ns.item_mapping = self.item_mapping

        ns.word_feature_shape = self.word_feature_shape

        return ns

    def check_interactions_in_folder(self) -> (t.Set[int], t.Set[int]):
        users = set()
        items = set()
        if self.users_vocabulary_features_path and self.items_vocabulary_features_path and self.users_tokens_path and self.items_tokens_path:
            with open(self.users_tokens_path, "r") as f:
                self.users_tokens = json.load(f)
                self.users_tokens = {int(k): v for k, v in self.users_tokens.items()}
            with open(self.items_tokens_path, "r") as f:
                self.items_tokens = json.load(f)
                self.items_tokens = {int(k): v for k, v in self.items_tokens.items()}
            users = users.union(list(self.users_tokens.keys()))
            items = items.union(list(self.items_tokens.keys()))
            self.users_word_features = np.load(self.users_vocabulary_features_path)
            self.items_word_features = np.load(self.items_vocabulary_features_path)
            self.word_feature_shape = self.users_word_features.shape[-1]
        if self.pos_users_path and self.pos_items_path:
            with open(self.pos_users_path, "r") as f:
                self.pos_users = json.load(f)
                self.pos_users = {int(k): v for k, v in self.pos_users.items()}
            with open(self.pos_items_path, "r") as f:
                self.pos_items = json.load(f)
                self.pos_items = {int(k): v for k, v in self.pos_items.items()}
        if users:
            self.user_mapping = {user: val for val, user in enumerate(users)}
        if items:
            self.item_mapping = {item: val for val, item in enumerate(items)}

        return users, items
