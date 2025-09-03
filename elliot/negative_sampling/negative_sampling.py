from signal import valid_signals

import pandas as pd
from types import SimpleNamespace
import typing as t
from scipy import sparse as sp
import numpy as np
import random
from ast import literal_eval as make_tuple

#from elliot.dataset.sparse_builder import SparseBuilder
from elliot.utils import sparse

np.random.seed(42)
random.seed(42)

"""
prefiltering:
    strategy: global_threshold|user_average|user_k_core|item_k_core|iterative_k_core|n_rounds_k_core|cold_users
    threshold: 3|average
    core: 5
    rounds: 2
"""

class NegativeSampler:

    def __init__(self, ns, public_users, public_items, private_users, private_items, i_train, val=None, test=None):
        self.ns = ns
        self.public_users = public_users
        self.public_items = public_items
        self.private_users = private_users
        self.private_items = private_items
        self.i_train = i_train
        self.val = val
        self.test = test

    def sample(self) -> t.Tuple[t.Optional[sp.csr_matrix], t.Optional[sp.csr_matrix]]:
        """Entry point: genera negative samples per validazione e test."""

        val_negative_items = None
        if self.val is not None:
            val_negative_items = self._process_sampling(validation=True)

        test_negative_items = None
        if self.test is not None:
            test_negative_items = self._process_sampling(validation=False)

        return val_negative_items, test_negative_items

    def _process_sampling(self, validation: bool = False) -> sp.csr_matrix:

        shape = (len(self.public_users), len(self.private_items))
        i_test = sparse.build_sparse(self.test, shape, self.public_users, self.public_items)

        candidate_negatives = np.where(((self.i_train + i_test).toarray() == 0), True, False)
        #all_true = SparseBuilder.create_sparse_matrix(np.ones(self.i_train.shape, dtype=bool), 'bool')
        #candidate_negatives = all_true - mask
        #del all_true, mask

        strategy = getattr(self.ns, "strategy", None)

        if strategy == "random":
            return self._random_strategy(candidate_negatives)

        elif strategy == "fixed":
            return self._fixed_strategy(validation)

        else:
            raise ValueError(f"Unsupported sampling strategy: {strategy}")

    def _random_strategy(self, candidate_negatives) -> sp.csr_matrix:
        num_items = getattr(self.ns, "num_items", None)
        #file_path = getattr(self.ns, "file_path", None)

        if not isinstance(num_items, int):
            raise ValueError("`num_items` must be an integer in negative_sampling config")

        negative_items = self._sample_by_random_uniform(candidate_negatives, num_items)

        #if file_path:
        #    self._save_to_file(negative_items, file_path)

        return negative_items

    def _fixed_strategy(self, validation: bool) -> sp.csr_matrix:
        files = getattr(self.ns, "files", None)
        if files is None:
            raise ValueError("Missing `files` option for fixed strategy")

        if not isinstance(files, list):
            files = [files]

        # Se validation=True, usa il secondo file
        file_ = files[1] if validation and len(files) > 1 else files[0]
        return self._read_from_files(file_)

    @staticmethod
    def _sample_by_random_uniform(data, num_items=99) -> sp.csr_matrix:
        """Campiona negativi per ogni utente da una matrice CSR di candidati."""
        rows, cols = [], []

        for row in range(data.shape[0]):
            mask = data[row]
            candidate_negatives = mask.nonzero()[0]

            if len(candidate_negatives) > num_items:
                sampled = np.random.choice(candidate_negatives, size=num_items, replace=False)
            else:
                sampled = candidate_negatives

            rows.append(np.full(sampled.shape, row, dtype=int))
            cols.append(sampled)

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        return sparse.build_sparse_mask(rows, cols, shape=data.shape)

    def _save_to_file(self, negative_items: sp.csr_matrix, file_path: str):
        """Salva i campioni negativi su file TSV."""
        nnz = negative_items.nonzero()
        with open(file_path, "w") as f:
            old_ind = 0
            for u, v in enumerate(negative_items.indptr[1:]):
                user_id = self.private_users[u]
                items = [self.private_items[i] for i in nnz[1][old_ind:v]]
                old_ind = v
                f.write(f"{(user_id,)}\t" + "\t".join(map(str, items)) + "\n")

    def _read_from_files(self, filepath: str) -> sp.csr_matrix:
        """Legge campioni negativi predefiniti da file."""
        map_ = {}
        with open(filepath) as file:
            for line in file:
                line = line.rstrip("\n").split('\t')
                user_id = self.public_users[int(make_tuple(line[0])[0])]
                int_set = {self.public_items[int(i)] for i in line[1:] if int(i) in self.public_items}
                map_[user_id] = int_set

        return sparse.build_sparse(map_, (len(self.public_users), len(self.public_items)),
                                   self.public_users, self.public_items, dtype=bool)


"""class NegativeSampler:

    @staticmethod
    def sample(ns: SimpleNamespace, public_users: t.Dict, public_items: t.Dict, private_users: t.Dict,
               private_items: t.Dict, i_train: sp.csr_matrix,
               val: t.Dict = None, test: t.Dict = None) -> t.Tuple[sp.csr_matrix, sp.csr_matrix]:

        val_negative_items = NegativeSampler._process_sampling(ns, public_users, public_items, private_users,
                                                              private_items, i_train,
                                                              test, validation=True) if val != None else None

        test_negative_items = NegativeSampler._process_sampling(ns, public_users, public_items, private_users,
                                                              private_items, i_train,
                                                               test) if test != None else None

        return (val_negative_items, test_negative_items) if val_negative_items else (test_negative_items, test_negative_items)

    @staticmethod
    def _process_sampling(ns: SimpleNamespace, public_users: t.Dict, public_items: t.Dict, private_users: t.Dict,
                         private_items: t.Dict, i_train: sp.csr_matrix,
                         test: t.Dict, validation=False) -> sp.csr_matrix:
        i_test = SparseBuilder.build_sparse(test, public_users, public_items)

        candidate_negatives = ((i_test + i_train).astype('bool') != True)
        ns = ns.negative_sampling

        strategy = getattr(ns, "strategy", None)

        if strategy == "random":
            num_items = getattr(ns, "num_items", None)
            file_path = getattr(ns, "file_path", None)
            if num_items is not None:
                if str(num_items).isdigit():
                    negative_items = NegativeSampler._sample_by_random_uniform(candidate_negatives, num_items)

                    nnz = negative_items.nonzero()
                    old_ind = 0
                    basic_negative = []
                    for u, v in enumerate(negative_items.indptr[1:]):
                        basic_negative.append([(private_users[u],), list(map(private_items.get, nnz[1][old_ind:v]))])
                        old_ind = v

                    with open(file_path, "w") as file:
                        for ele in basic_negative:
                            line = str(ele[0]) + '\t' + '\t'.join(map(str, ele[1]))+'\n'
                            file.write(line)

                    pass
                else:
                    raise Exception("Number of negative items value not recognized")
            else:
                raise Exception("Number of negative items option is missing")
        elif strategy == "fixed":
            files = getattr(ns, "files", None)
            if files is not None:
                if not isinstance(files, list):
                    files = [files]
                file_ = files[0] if validation == False else files[1]
                negative_items = NegativeSampler._read_from_files(public_users, public_items, file_)
            pass
        else:
            raise Exception("Missing strategy")

        return negative_items

    @staticmethod
    def _sample_by_random_uniform(data: sp.csr_matrix, num_items=99) -> sp.csr_matrix:
        rows = []
        cols = []
        for row in range(data.shape[0]):
            candidate_negatives = list(zip(*data.getrow(row).nonzero()))
            sampled_negatives = np.array(candidate_negatives)[random.sample(range(len(candidate_negatives)), num_items)]
            rows.extend(list(np.ones(len(sampled_negatives), dtype=int) * row))
            cols.extend(sampled_negatives[:, 1])
        indptr = data.indptr
        indices = data.indices

        for row in range(data.shape[0]):
            start, end = indptr[row], indptr[row + 1]
            candidate_negatives = indices[start:end]

            if len(candidate_negatives) > num_items:
                sampled = np.random.choice(candidate_negatives, size=num_items, replace=False)
            else:
                sampled = candidate_negatives

            rows.append(np.full(sampled.shape, row, dtype=int))
            cols.append(sampled)

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        negative_samples = SparseBuilder.create_sparse_matrix(
            (np.ones_like(rows), (rows, cols)), dtype='bool', shape=data.shape
        )
        return negative_samples

    @staticmethod
    def _read_from_files(public_users: t.Dict, public_items: t.Dict, filepath: str) -> sp.csr_matrix:

        map_ = {}
        with open(filepath) as file:
            for line in file:
                line = line.rstrip("\n").split('\t')
                int_set = {public_items[int(i)] for i in line[1:] if int(i) in public_items.keys()}
                map_[public_users[int(make_tuple(line[0])[0])]] = int_set

        #rows_cols = [(u, i) for u, items in map_.items() for i in items]
        #rows, cols = zip(*rows_cols)
        # rows = [u for u, _ in rows_cols]
        # cols = [i for _, i in rows_cols]
        negative_samples = SparseBuilder.build_sparse(map_, public_users, public_items, dtype='bool')
        return negative_samples

    @staticmethod
    def build_sparse(map_ : t.Dict, nusers: int, nitems: int):

        rows_cols = [(u, i) for u, items in map_.items() for i in items.keys()]
        rows = [u for u, _ in rows_cols]
        cols = [i for _, i in rows_cols]
        data = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                             shape=(nusers, nitems))
        return data"""
