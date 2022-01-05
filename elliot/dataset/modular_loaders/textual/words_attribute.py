import typing as t
import pandas as pd
import numpy as np
from ast import literal_eval
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class WordsTextualAttributes(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.vocabulary_features_path = getattr(ns, "vocabulary_features", None)
        self.train_reviews_tokens_path = getattr(ns, "train_reviews_tokens", None)

        self.item_mapping = {}
        self.user_mapping = {}
        self.word_feature_shape = None
        self.word_features = None
        self.all_reviews_tokens = None

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
        ns.vocabulary_features_path = self.vocabulary_features_path
        ns.train_reviews_tokens_path = self.train_reviews_tokens_path

        ns.user_mapping = self.user_mapping
        ns.item_mapping = self.item_mapping

        ns.word_feature_shape = self.word_feature_shape

        return ns

    def check_interactions_in_folder(self) -> (t.Set[int], t.Set[int]):
        users = set()
        items = set()
        if self.vocabulary_features_path and self.train_reviews_tokens_path:
            self.all_reviews_tokens = pd.read_csv(self.train_reviews_tokens_path, sep='\t')
            self.all_reviews_tokens['tokens_position'] = self.all_reviews_tokens['tokens_position'].apply(
                lambda x: literal_eval(x))
            users = users.union(self.all_reviews_tokens['USER_ID'].unique().tolist())
            items = items.union(self.all_reviews_tokens['ITEM_ID'].unique().tolist())
            self.word_features = np.load(self.vocabulary_features_path)
            self.word_feature_shape = self.word_features.shape
        if users:
            self.user_mapping = {user: val for val, user in enumerate(users)}
        if items:
            self.item_mapping = {item: val for val, item in enumerate(items)}

        return users, items
