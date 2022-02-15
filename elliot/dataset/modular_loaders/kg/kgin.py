from types import SimpleNamespace
import typing as t
import numpy as np
import pandas as pd

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class KGINLoader(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.attribute_file = getattr(ns, "attributes", None)
        self.entities_file = getattr(ns, "entities", None)
        self.users = users
        self.items = items

        if self.attribute_file is not None:
            self.map_ = self.read_triplets(self.attribute_file)
            # self.items = self.items & set(self.map_.keys())

        self.entities = set()

        with open(self.entities_file) as f:
            # next(f)     # considers the header
            for line in f:
                self.entities.add(int(line.split(' ')[-1]))
                # self.entities.add(int(line.split('\n')[-1]))

        # TODO: in realtà sarebbe interessante capire quali item sono stati eliminati, per rimuoverli anche da entities
        self.entity_list = set.difference(self.entities, self.items)

    def get_mapped(self):
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items
        self.map_ = self.map_[np.where(np.isin(self.map_[:, 0], list(self.items)))]
        self.map_ = self.map_[np.where(np.isin(self.map_[:, 2], list(self.items) + list(self.entity_list)))]
        # self.items = self.items & set(self.map_.keys())

    def create_namespace(self):
        ns = SimpleNamespace()
        ns.__name__ = "KGINLoader"
        ns.object = self
        ns.__dict__.update(self.__dict__)
        ns.feature_map = self.map_
        ns.relations = np.unique(ns.feature_map[:, 1])
        ns.n_relations = len(ns.relations) + 1
        # ns.entities = np.unique(ns.feature_map[:, 2])
        ns.n_entities = len(self.items) + len(ns.entity_list)
        ns.n_nodes = ns.n_entities + len(self.users)
        ns.private_relations = {p[0] + 1: f for p, f in list(np.ndenumerate(ns.relations))}
        ns.public_relations = {v: k for k, v in ns.private_relations.items()}
        ns.private_objects = {p + len(self.items): f for p, f in list(enumerate(ns.entity_list))}
        ns.public_objects = {v: k for k, v in ns.private_objects.items()}
        return ns

    def read_triplets(self, file_name):
        can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
        can_triplets_np = np.unique(can_triplets_np, axis=0)

        # # get triplets with inverse direction like <entity, is-aspect-of, item>
        # inv_triplets_np = can_triplets_np.copy()
        # inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        # inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        # inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # # consider two additional relations --- 'interact' and 'be interacted'
        # can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        # inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # # get full version of knowledge graph
        # triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)

        # consider two additional relations --- 'interact'.
        triplets = can_triplets_np.copy()

        return triplets