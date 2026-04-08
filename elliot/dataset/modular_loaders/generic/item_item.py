from typing import Tuple, Set
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
from elliot.utils.registry import side_info_registry


@side_info_registry.register(
    name="ItemItem",
    provides="item_features",
    format="sparse"
)
class ItemItem(AbstractLoader):
    interactions_ii: str

    def __init__(self, **params):
        super().__init__(**params)
        self.item_mapping = {}
        self.user_mapping = {}

    def get_mapped(self) -> Tuple[Set[int], Set[int]]:
        return self.users, self.items

    def filter(self, users: Set[int], items: Set[int]):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = self.name
        ns.object = self
        return ns

    def get_all_features(self, public_items):
        int_sim = self.reader.read_json(self.interactions_ii)

        rows_ii, cols_ii = [], []
        for k, v in int_sim.items():
            for val in v:
                rows_ii.append(public_items[k if not k.isdigit() else int(k)])
                cols_ii.append(public_items[val])

        return rows_ii, cols_ii
