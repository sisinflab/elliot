from types import SimpleNamespace
import typing as t
from os.path import splitext

import numpy as np
import pandas as pd

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class KGFlexLoader(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.mapping_path = getattr(ns, "mapping", None)
        self.train_path = getattr(ns, "kg_train", None)
        self.dev_path = getattr(ns, "kg_dev", None)
        self.test_path = getattr(ns, "kg_test", None)
        self.second_hop_path = getattr(ns, "second_hop", None)
        self.properties_file = getattr(ns, "properties", None)
        self.additive = getattr(ns, "additive", True)
        self.threshold = getattr(ns, "threshold", 10)
        self.users = users
        self.items = items

        self.mapping = self.load_mapping_file(self.mapping_path)
        self.properties = self.load_properties(self.properties_file)
        train_triples = pd.read_csv(self.train_path, sep='\t', names=['uri', 'predicate', 'object'],
                                    dtype={'uri': str, 'predicate': str, 'object': str})
        self.dev_triples = None
        if self.dev_path:
            self.dev_triples = pd.read_csv(self.dev_path, sep='\t', names=['uri', 'predicate', 'object'],
                                    dtype={'uri': str, 'predicate': str, 'object': str})
        self.test_triples = None
        if self.test_path:
            self.test_triples = pd.read_csv(self.test_path, sep='\t', names=['uri', 'predicate', 'object'],
                                    dtype={'uri': str, 'predicate': str, 'object': str})
        self.triples = pd.concat([train_triples, self.dev_triples, self.test_triples])
        del train_triples, self.dev_triples, self.test_triples

        self.second_hop = pd.DataFrame(columns=['uri', 'predicate', 'object'])\
            .astype(dtype={'uri': str, 'predicate': str, 'object': str})
        if self.second_hop_path:
            import time
            if self.second_hop_path.endswith("tar.gz"):
                start = time.time()
                self.second_hop = pd.read_csv(self.second_hop_path, compression='gzip', sep='\t', names=['uri', 'predicate', 'object'],
                                    dtype={'uri': str, 'predicate': str, 'object': str})
                self.logger.info(f"Time taken to load Second Hop: {time.time() - start}")
            elif self.second_hop_path.endswith("tar.bz2"):
                start = time.time()
                self.second_hop = pd.read_csv(self.second_hop_path, compression='bz2', sep='\t', names=['uri', 'predicate', 'object'],
                                    dtype={'uri': str, 'predicate': str, 'object': str})
                self.logger.info(f"Time taken to load Second Hop: {time.time() - start}")
            elif self.second_hop_path.endswith("tar.xz"):
                start = time.time()
                self.second_hop = pd.read_csv(self.second_hop_path, compression='xz', sep='\t', names=['uri', 'predicate', 'object'],
                                    dtype={'uri': str, 'predicate': str, 'object': str})
                self.logger.info(f"Time taken to load Second Hop: {time.time() - start}")
            else:
                start = time.time()
                self.second_hop = pd.read_csv(self.second_hop_path, sep='\t', names=['uri', 'predicate', 'object'],
                                              dtype={'uri': str, 'predicate': str, 'object': str})
                self.logger.info(f"Time taken to load Second Hop: {time.time() - start}")

        if self.properties:
            if self.additive:
                self.triples = self.triples[self.triples["predicate"].isin(self.properties)]
                self.second_hop = self.second_hop[self.second_hop["predicate"].isin(self.properties)]
            else:
                self.triples = self.triples[~self.triples["predicate"].isin(self.properties)]
                self.second_hop = self.second_hop[~self.second_hop["predicate"].isin(self.properties)]

        # COMPUTE FEATURES
        occurrences_per_feature = self.triples.groupby(['predicate', 'object']).size().to_dict()
        keep_set = {f for f, occ in occurrences_per_feature.items() if occ > self.threshold}

        second_order_features = self.triples.merge(self.second_hop, left_on='object', right_on='uri', how='left')
        second_order_features = second_order_features[second_order_features['uri_y'].notna()]
        occurrences_per_feature_2 = second_order_features.groupby(['predicate_x', 'predicate_y', 'object_y']) \
            .size().to_dict()
        keep_set2 = {f for f, occ in occurrences_per_feature_2.items() if occ > self.threshold}

        self.triples = self.triples[
            self.triples[['predicate', 'object']].set_index(['predicate', 'object']).index.map(
                lambda f: f in keep_set)].astype(str)

        if len(second_order_features) > 0:
            self.second_order_features = second_order_features[second_order_features[
                ['predicate_x', 'predicate_y', 'object_y']].set_index(['predicate_x', 'predicate_y', 'object_y'])
                .index.map(lambda f: f in keep_set2)].astype(str)
            #self.second_order_features = self.second_order_features.drop(['object_x', 'uri_y'], axis=1)
            self.second_order_features = self.second_order_features.drop(['uri_y'], axis=1)
        else:
            self.second_order_features = pd.DataFrame(
                columns=['uri_x', 'predicate_x', 'object_x', 'predicate_y', 'object_y']).astype(
                dtype={'uri_x': str, 'predicate_x': str, 'object_x': str, 'predicate_y': str, 'object_y': str})

        possible_items = [str(uri) for uri in self.triples["uri"].unique()]
        self.mapping = {k: v for k, v in self.mapping.items() if v in possible_items}
        self.items = self.items & set(self.mapping.keys())

    def get_mapped(self):
        return self.users, self.items

    def filter(self, users, items):
        self.users = self.users & users
        self.mapping = {k: v for k, v in self.mapping.items() if k in items}
        self.items = {i for i in self.items if i in self.mapping.keys()}

    def create_namespace(self):
        ns = SimpleNamespace()
        ns.__name__ = "KGFlexLoader"
        ns.object = self
        ns.__dict__.update(self.__dict__)
        return ns

    def load_properties(self, properties_file):
        properties = []
        if properties_file:
            with open(properties_file) as file:
                for line in file:
                    if line[0] != '#':
                        properties.append(line.rstrip("\n"))
        return properties

    def read_triples(self, path: str) -> t.List[t.Tuple[str, str, str]]:
        triples = []

        tmp = splitext(path)
        ext = tmp[1] if len(tmp) > 1 else None

        with open(path, 'rt') as f:
            for line in f.readlines():
                if ext is not None and ext.lower() == '.tsv':
                    s, p, o = line.split('\t')
                else:
                    s, p, o = line.split()
                triples += [(s.strip(), p.strip(), o.strip())]
        return triples

    def triples_to_vectors(self, triples: t.List[t.Tuple[str, str, str]],
                           entity_to_idx: t.Dict[str, int],
                           predicate_to_idx: t.Dict[str, int]) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Xs = np.array([entity_to_idx[s] for (s, p, o) in triples], dtype=np.int32)
        Xp = np.array([predicate_to_idx[p] for (s, p, o) in triples], dtype=np.int32)
        Xo = np.array([entity_to_idx[o] for (s, p, o) in triples], dtype=np.int32)
        return Xs, Xp, Xo

    def load_mapping_file(self, mapping_file, separator='\t'):
        map = {}
        with open(mapping_file) as file:
            for line in file:
                line = line.rstrip("\n").split(separator)
                map[int(line[0])] = line[1]
        return map
