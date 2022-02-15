from types import SimpleNamespace
import typing as t
import numpy as np
import pandas as pd

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class KGINTSVLoader(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.attribute_file = getattr(ns, "attributes", None)
        self.entities_file = getattr(ns, "entities", None)
        self.users = users
        self.items = items

        if self.attribute_file is not None:
            self.map_ = self.read_triplets(self.attribute_file)
            # self.items = self.items & set(self.map_.keys())

        entities = pd.read_csv(self.entities_file, sep='\t', header=0, names=['id', 'remap'])
        self.entities = set(entities.remap.unique())

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
        ns.__name__ = "KGINTSVLoader"
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
        kg = pd.read_csv(file_name, sep='\t', header=None)
        return np.array(kg)
