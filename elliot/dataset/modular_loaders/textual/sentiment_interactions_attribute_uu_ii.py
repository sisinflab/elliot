import typing as t
import json
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class SentimentInteractionsTextualAttributesUUII(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.interactions_sim_uu = getattr(ns, "interactions_sim_uu", None)
        self.interactions_sim_ii = getattr(ns, "interactions_sim_ii", None)

        self.item_mapping = {}
        self.user_mapping = {}

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
        ns.__name__ = "SentimentInteractionsTextualAttributesUUII"
        ns.object = self

        return ns

    def check_interactions_in_folder(self) -> (t.Set[int], t.Set[int]):
        users = set()
        items = set()
        if self.interactions_sim_uu:
            with open(self.interactions_sim_uu, 'r') as f:
                int_sim = json.load(f)
            users_from_json = [[int(k.split('_')[0]), int(k.split('_')[1])] for k in int_sim]
            users_from_json = set([user for sublist in users_from_json for user in sublist])
            users = users.union(users_from_json)

        if self.interactions_sim_ii:
            with open(self.interactions_sim_ii, 'r') as f:
                int_sim = json.load(f)
            items_from_json = [[int(k.split('_')[0]), int(k.split('_')[1])] for k in int_sim]
            items_from_json = set([item for sublist in items_from_json for item in sublist])
            items = items.union(items_from_json)

        if users:
            self.user_mapping = {user: val for val, user in enumerate(users)}

        if items:
            self.item_mapping = {item: val for val, item in enumerate(items)}

        return users, items

    def get_all_features(self, public_users, public_items):

        def add_zero(s):
            zeros_to_add = num_digits - len(s)
            return ''.join(['0' for _ in range(zeros_to_add)]) + s

        def remove_zero(s):
            return int(s)

        with open(self.interactions_sim_uu, 'r') as f:
            int_sim = json.load(f)
        all_interactions_uu = list(int_sim.values())
        private_keys = list(int_sim.keys())
        num_digits = len(str(int(list(public_users.keys())[-1])))
        indices_public = ['_'.join(
            [add_zero(str(public_users[float(inter.split('_')[0])])),
             add_zero(str(public_users[float(inter.split('_')[1])]))]) for inter in
            private_keys]
        sorted_indices_public = sorted(range(len(indices_public)), key=lambda k: indices_public[k])
        all_interactions_uu = [all_interactions_uu[index] for index in sorted_indices_public]
        sorted_indices_public = sorted(indices_public)
        sorted_indices_public = [(remove_zero(s.split('_')[0]), remove_zero(s.split('_')[1])) for s in
                                 sorted_indices_public]
        rows_uu, cols_uu = list(map(list, zip(*sorted_indices_public)))

        with open(self.interactions_sim_ii, 'r') as f:
            int_sim = json.load(f)
        all_interactions_ii = list(int_sim.values())
        private_keys = list(int_sim.keys())
        num_digits = len(str(int(list(public_items.keys())[-1])))
        indices_public = ['_'.join(
            [add_zero(str(public_items[float(inter.split('_')[0])])),
             add_zero(str(public_items[float(inter.split('_')[1])]))]) for inter in
            private_keys]
        sorted_indices_public = sorted(range(len(indices_public)), key=lambda k: indices_public[k])
        all_interactions_ii = [all_interactions_ii[index] for index in sorted_indices_public]
        sorted_indices_public = sorted(indices_public)
        sorted_indices_public = [(remove_zero(s.split('_')[0]), remove_zero(s.split('_')[1])) for s in
                                 sorted_indices_public]
        rows_ii, cols_ii = list(map(list, zip(*sorted_indices_public)))
        return all_interactions_uu, all_interactions_ii, rows_uu, rows_ii, cols_uu, cols_ii
