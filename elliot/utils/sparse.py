import numpy as np
from scipy import sparse as sp


def build_sparse(data_dict, shape, public_users=None, public_items=None, dtype='float32') -> sp.csr_matrix:
    #rows_cols = [(u, i) for u, items in dict.items() for i in items.keys()]
    rows, cols = _extract_user_item_pairs(data_dict, public_users, public_items)

    return create_sparse_matrix((np.ones_like(rows), (rows, cols)), dtype=dtype, shape=shape)


def build_sparse_ratings(data_dict, shape, public_users=None, public_items=None, dtype='float32') -> sp.csr_matrix:
    #rows_cols_ratings = [(u, i, r) for u, items in dict.items() for i, r in items.items()]
    rows, cols, ratings = _extract_user_item_rating_triples(data_dict)

    return create_sparse_matrix((ratings, (rows, cols)), dtype=dtype, shape=shape)


def build_sparse_mask(rows, cols, shape):
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)

    n_total = shape[0] * shape[1]
    n_true = len(rows)
    density = n_true / n_total

    if density <= 0.5:
        # normale: True nei punti specificati
        data = np.ones(len(rows), dtype=bool)
        mask = create_sparse_matrix((data, (rows, cols)), dtype=bool, shape=shape)
        return mask, False
    else:
        # più conveniente memorizzare i False
        dense_bool = np.ones(shape, dtype=bool)
        dense_bool[rows, cols] = False
        inv_mask = create_sparse_matrix(dense_bool, dtype=bool)
        return inv_mask, True


def create_sparse_matrix(data, dtype, shape=None) -> sp.csr_matrix:
    if shape is None:
        if isinstance(data, (np.ndarray, sp.csr_matrix)):
            shape = data.shape
        else:
            raise ValueError("Parameter 'data' must be either a numpy array or a CSR sparse matrix"
                             "if 'shape' is not specified")
    sparse_data = sp.csr_matrix(data, dtype=dtype, shape=shape)
    return sparse_data


def _extract_user_item_pairs(data_dict, public_users=None, public_items=None):
    if public_users is not None and public_items is not None:
        #rows_cols = [(public_users[user], public_items[i])
        #             for user, items in data_dict.items() if user in public_users.keys()
        #             for i in items.keys() if i in public_items.keys()]
        pu_get = public_users.get
        pi_get = public_items.get
        pi_keys = set(public_items)

        rows_cols = [
            (u_idx, pi_get(i))
            for user, items in data_dict.items()
            if (u_idx := pu_get(user)) is not None
            for i in items.keys() & pi_keys
        ]
    else:
        rows_cols = [(u, i) for u, items in data_dict.items() for i in items.keys()]
    rows = [u for u, _ in rows_cols]
    cols = [i for _, i in rows_cols]
    return rows, cols


def _extract_user_item_rating_triples(data_dict):
    rows_cols_ratings = [(u, i, r) for u, items in data_dict.items() for i, r in items.items()]
    rows = [u for u, _, _ in rows_cols_ratings]
    cols = [i for _, i, _ in rows_cols_ratings]
    ratings = [r for _, _, r in rows_cols_ratings]
    return rows, cols, ratings


def _check_users_items(users, items):
    if isinstance(users, list) and isinstance(items, list):
        return None, None
    elif isinstance(users, dict) and isinstance(items, dict):
        return users, items
    else:
        raise ValueError("Invalid type for parameters 'users' or 'items'")


"""def sparse_not(A: sp.csr_matrix) -> sp.csr_matrix:
    if A.dtype != bool:
        raise ValueError("La matrice deve avere dtype=bool")

    n_rows, n_cols = A.shape
    rows, cols = [], []

    for row in range(n_rows):
        start, end = A.indptr[row], A.indptr[row + 1]
        positives = A.indices[start:end]  # colonne con True
        negatives = np.setdiff1d(np.arange(n_cols), positives, assume_unique=True)

        if len(negatives) > 0:
            rows.append(np.full(len(negatives), row, dtype=int))
            cols.append(negatives)

    #if not rows:
    #    return sp.csr_matrix(A.shape, dtype=bool)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    return create_sparse_matrix(
        (np.ones_like(rows, dtype=bool), (rows, cols)), shape=A.shape, dtype='bool'
    )"""
