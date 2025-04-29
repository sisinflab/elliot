import numpy as np
from scipy import sparse as sp


class SparseBuilder:

    @staticmethod
    def build_sparse(dict, users, items, dtype='float32') -> sp.csr_matrix:
        rows_cols = [(u, i) for u, items in dict.items() for i in items.keys()]
        rows, cols = map(list, zip(*rows_cols))

        return SparseBuilder.create_sparse_matrix(
            rows, cols, np.ones_like(rows), len(users), len(items), dtype
        )

    @staticmethod
    def build_sparse_ratings(dict, users, items, dtype='float32') -> sp.csr_matrix:
        rows_cols_ratings = [(u, i, r) for u, items in dict.items() for i, r in items.items()]
        rows, cols, ratings = map(list, zip(*rows_cols_ratings))

        return SparseBuilder.create_sparse_matrix(
            rows, cols, ratings, len(users), len(items), dtype
        )

    @staticmethod
    def build_sparse_public(dict, users, items, dtype='float32') -> sp.csr_matrix:
        i_test = [(users[user], items[i])
                  for user, items in dict.items() if user in users.keys()
                  for i in items.keys() if i in items.keys()]
        rows, cols = map(list, zip(*i_test))

        return SparseBuilder.create_sparse_matrix(
            rows, cols, np.ones_like(rows), len(users), len(items), dtype
        )

    @staticmethod
    def create_sparse_matrix(rows, cols, data, n_users, n_items, dtype) -> sp.csr_matrix:
        sparse_data = sp.csr_matrix((data, (rows, cols)), dtype=dtype, shape=(n_users, n_items))
        return sparse_data
