from types import SimpleNamespace
import typing as t
import numpy as np

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class KGINLoader(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.attribute_file = getattr(ns, "attributes", None)
        self.users = users
        self.items = items

        if self.attribute_file is not None:
            self.map_ = self.read_triplets(self.attribute_file)
            # self.items = self.items & set(self.map_.keys())

    def get_mapped(self):
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        pass

    def create_namespace(self):
        ns = SimpleNamespace()
        ns.__name__ = "KGINLoader"
        ns.object = self
        ns.feature_map = self.map_
        ns.features = list({f for i in self.items for f in ns.feature_map[i]})
        ns.nfeatures = len(ns.features)
        ns.private_features = {p: f for p, f in enumerate(ns.features)}
        ns.public_features = {v: k for k, v in ns.private_features.items()}
        return ns

    def read_triplets(self, file_name):
        can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
        can_triplets_np = np.unique(can_triplets_np, axis=0)

        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)

        return triplets