from types import SimpleNamespace
import typing as t
from os.path import splitext

import numpy as np

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class KGCompletion(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.train_path = getattr(ns, "train_path", None)
        self.dev_path = getattr(ns, "dev_path", None)
        self.test_path = getattr(ns, "test_path", None)
        self.test_i_path = getattr(ns, "test_i_path", None)
        self.test_ii_path = getattr(ns, "test_ii_path", None)
        self.input_type = getattr(ns, "input_type", "standard")
        self.users = users
        self.items = items

        assert self.input_type in {'standard', 'reciprocal'}

        self.Xi = self.Xs = self.Xp = self.Xo = None

        # Loading the dataset
        self.train_triples = self.read_triples(self.train_path) if self.train_path else []
        self.original_predicate_names = {p for (_, p, _) in self.train_triples}

        self.reciprocal_train_triples = None
        if self.input_type in {'reciprocal'}:
            self.reciprocal_train_triples = [(o, f'inverse_{p}', s) for (s, p, o) in self.train_triples]
            self.train_triples += self.reciprocal_train_triples

        self.dev_triples = self.read_triples(self.dev_path) if self.dev_path else []
        self.test_triples = self.read_triples(self.test_path) if self.test_path else []

        self.test_i_triples = self.read_triples(self.test_i_path) if self.test_i_path else []
        self.test_ii_triples = self.read_triples(self.test_ii_path) if self.test_ii_path else []

        self.all_triples = self.train_triples + self.dev_triples + self.test_triples

        self.entity_set = {s for (s, _, _) in self.all_triples} | {o for (_, _, o) in self.all_triples}
        self.predicate_set = {p for (_, p, _) in self.all_triples}

        self.nb_examples = len(self.train_triples)

        self.entity_to_idx = {entity: idx for idx, entity in enumerate(sorted(self.entity_set))}
        self.nb_entities = max(self.entity_to_idx.values()) + 1
        self.idx_to_entity = {v: k for k, v in self.entity_to_idx.items()}

        self.predicate_to_idx = {predicate: idx for idx, predicate in enumerate(sorted(self.predicate_set))}
        self.nb_predicates = max(self.predicate_to_idx.values()) + 1
        self.idx_to_predicate = {v: k for k, v in self.predicate_to_idx.items()}

        self.inverse_of_idx = {}
        if self.input_type in {'reciprocal'}:
            for p in self.original_predicate_names:
                p_idx, ip_idx = self.predicate_to_idx[p], self.predicate_to_idx[f'inverse_{p}']
                self.inverse_of_idx.update({p_idx: ip_idx, ip_idx: p_idx})

        # Triples
        self.Xs, self.Xp, self.Xo = self.triples_to_vectors(self.train_triples, self.entity_to_idx, self.predicate_to_idx)
        self.Xi = np.arange(start=0, stop=self.Xs.shape[0], dtype=np.int32)

        self.dev_Xs, self.dev_Xp, self.dev_Xo = self.triples_to_vectors(self.dev_triples, self.entity_to_idx,
                                                                   self.predicate_to_idx)
        self.dev_Xi = np.arange(start=0, stop=self.dev_Xs.shape[0], dtype=np.int32)

        assert self.Xs.shape == self.Xp.shape == self.Xo.shape == self.Xi.shape
        assert self.dev_Xs.shape == self.dev_Xp.shape == self.dev_Xo.shape == self.dev_Xi.shape

    def get_mapped(self):
        return self.users, self.items

    def filter(self, users, items):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self):
        ns = SimpleNamespace()
        ns.__name__ = "KGCompletion"
        ns.object = self
        ns.__dict__.update(self.__dict__)
        return ns

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
