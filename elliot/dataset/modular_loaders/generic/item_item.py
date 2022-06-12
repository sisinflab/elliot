import typing as t
import json
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class ItemItem(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.interactions_ii = getattr(ns, "interactions", None)

        self.item_mapping = {}
        self.user_mapping = {}

        self.users = users
        self.items = items

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = "ItemItem"
        ns.object = self

        return ns

    def get_all_features(self, public_items):
        rows_ii, cols_ii = [], []
        with open(self.interactions_ii, 'r') as f:
            int_sim = json.load(f)

        for k, v in int_sim.items():
            for val in v:
                rows_ii.append(public_items[k if not k.isdigit() else int(k)])
                cols_ii.append(public_items[val])

        return rows_ii, cols_ii
