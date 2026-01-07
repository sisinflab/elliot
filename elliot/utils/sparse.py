from typing import Union
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


def zero_intervals(n_cols, nnz_sorted):
    intervals = []
    prev = -1
    for c in nnz_sorted:
        if c > prev + 1:
            intervals.append((prev + 1, c - 1))
        prev = c
    if prev < n_cols - 1:
        intervals.append((prev + 1, n_cols - 1))
    return intervals


def center_data(
    R: Union[csr_matrix, csc_matrix],
    axis: int = 0,
    copy: bool = True
) -> csr_matrix:

    if axis == 0:
        M = R.tocsc(copy=copy)
    else:
        M = R.tocsr(copy=copy)

    sums = np.asarray(M.sum(axis=axis)).ravel()
    nnz = np.diff(M.indptr)

    means = np.zeros_like(sums)
    mask = nnz > 0
    means[mask] = sums[mask] / nnz[mask]

    expanded_means = np.repeat(means, nnz)
    M.data -= expanded_means

    return M.tocsr()
