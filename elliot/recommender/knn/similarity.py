import logging as pylog
import numpy as np
import similaripy as sim
from scipy.sparse import csr_matrix
from tqdm import tqdm
from sklearn.metrics.pairwise import chi2_kernel, pairwise_distances_chunked

from elliot.utils import logging as elog
from elliot.utils.sparse import center_data


class Similarity(object):
    SUPPORTED_SIMILARITIES = {"cosine", "dot", "asym", "jaccard", "dice", "tversky", "rp3beta"}
    SUPPORTED_DISSIMILARITIES = {
        "euclidean", "manhattan", "haversine", "cityblock", "l1", "l2", #"chi2",
        "braycurtis", "canberra", "chebyshev", "correlation", "hamming", #"kulsinski", "mahalanobis",
        "minkowski", "rogerstanimoto", "russellrao", "seuclidean",
        "sokalmichener", "sokalsneath", "sqeuclidean", "yule"
    }

    def __init__(
        self,
        train_data,
        item_profile=None,
        user_profile=None,
        similarity='cosine',
        num_neighbors=-1,
        asymmetric_alpha=0.5,
        alpha=1.0,
        beta=1.0
    ):
        if item_profile is not None and user_profile is not None:
            self.X, self.Y = user_profile, item_profile
        else:
            self.X = (item_profile or user_profile or train_data)
            self.Y = None

        self.train_data = train_data

        self.similarity = similarity
        self.asym_alpha = asymmetric_alpha
        self.alpha = alpha
        self.beta = beta

        self.tot_neighbors = train_data.shape[0]
        self.num_neighbors = num_neighbors if num_neighbors < self.tot_neighbors else -1

        self.logger = elog.get_logger(self.__class__.__name__)

    def compute_similarity(self):
        """Compute similarity or distance-based similarity matrix."""
        self.logger.info(
            "Supported similarities and distances",
            extra={"context": {
                "similarities": sorted(self.SUPPORTED_SIMILARITIES),
                "dissimilarities": sorted(self.SUPPORTED_DISSIMILARITIES)
            }}
        )

        if self.similarity not in (self.SUPPORTED_SIMILARITIES | self.SUPPORTED_DISSIMILARITIES):
            raise ValueError(
                f"Similarity: similarity '{self.similarity}' not recognized.\n"
                f"Allowed values: {self.SUPPORTED_SIMILARITIES | self.SUPPORTED_DISSIMILARITIES}"
            )

        if self.similarity in self.SUPPORTED_SIMILARITIES:
            return self._compute_direct_similarity()
        else:
            return self._compute_distance_based_similarity()

    def _compute_direct_similarity(self):
        """Similarities natively supported by Similaripy."""
        k = self.num_neighbors if self.num_neighbors > -1 else self.tot_neighbors

        match self.similarity:
            case "cosine":
                w_mat = sim.cosine(
                    self.X, self.Y, k=k, format_output='csr'
                )

            case "asym":
                w_mat = sim.asymmetric_cosine(
                    self.X, self.Y, k=k, alpha=self.asym_alpha, format_output='csr'
                )

            case "dot":
                w_mat = sim.dot_product(
                    self.train_data, k=k, format_output='csr'
                )

            case "jaccard":
                w_mat = sim.jaccard(
                    self.X, self.Y, k=k, binary=True, format_output='csr'
                )

            case "dice":
                w_mat = sim.dice(
                    self.X, self.Y, k=k, binary=True, format_output='csr'
                )

            case "tversky":
                w_mat = sim.tversky(
                    self.X, self.Y, k=k, alpha=self.alpha, beta=self.beta, binary=True, format_output='csr'
                )

            case "rp3beta":
                w_mat = sim.rp3beta(
                    self.X, self.Y, k=k, alpha=self.alpha, beta=self.beta, binary=True, format_output='csr'
                )

            # case "spcorrelation":
            #     centered_X = center_data(self.X, axis=1)
            #     centered_Y = center_data(self.Y, axis=0) if self.Y is not None else None
            #     w_mat = sim.cosine(
            #         centered_X, centered_Y, k=self.num_neighbors, format_output='csr'
            #     )

        return w_mat

    def _compute_distance_based_similarity(self):
        """Distances from sklearn."""
        sparse_ok = {'euclidean', 'manhattan', 'haversine', 'chi2', 'cityblock', 'l1', 'l2'}
        X, Y = (var.astype(np.float32) if var is not None else None for var in (self.X, self.Y))
        X, Y = (X, Y) if self.similarity in sparse_ok else (X.toarray(), None if Y is None else Y.toarray())

        #if self.similarity == 'chi2':
        #    dist_matrix = chi2_kernel(X, Y)
        #else:
        dist_matrix = pairwise_distances_chunked(X, Y, metric=self.similarity)

        return self._process_similarity(dist_matrix)

    def _process_similarity(self, dist_matrix):
        """Compute similarity matrix from distance matrix, keeping top-k neighbors."""
        if not self.num_neighbors > -1:
            return dist_matrix

        data, cols_indices, row_indptr = [], [], [0]
        k = self.num_neighbors
        processed_rows = 0

        t = tqdm(total=None)
        t.set_description("Computing")

        for dist_chunk in dist_matrix:
            chunk = 1 / (1 + dist_chunk)

            idx = np.argpartition(chunk, -k, axis=1)[:, -k:]
            top_vals = np.take_along_axis(chunk, idx, axis=1)

            data.extend(top_vals.ravel())
            cols_indices.extend(idx.ravel())

            last = row_indptr[-1]
            new_ptrs = np.arange(last + k, last + k * (chunk.shape[0] + 1), k)
            row_indptr.extend(new_ptrs.tolist())

            processed_rows += chunk.shape[0]
            t.update(1)
            t.set_postfix(rows=processed_rows)

        t.set_description("Build csr matrix")

        sim_matrix = csr_matrix(
            (data, cols_indices, row_indptr),
            shape=(self.tot_neighbors, self.tot_neighbors),
            dtype=np.float32
        )

        t.set_description("Done")
        t.refresh()

        return sim_matrix


"""def chi2_kernel_sparse_chunked(X, Y=None, gamma=1.0, batch_size=512):
    if Y is None:
        Y = X

    if not sparse.isspmatrix_csr(X) or not sparse.isspmatrix_csr(Y):
        raise TypeError("X and Y must be of type 'scipy.sparse.csr_matrix'")

    working_memory = 512
    bytes_per_element = np.dtype(np.float32).itemsize
    max_bytes = working_memory * (2 ** 20)
    batch_size = max(1, int(max_bytes // (bytes_per_element * n_features * n_Y)))

    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]

    iterator = range(0, n_samples_X, batch_size)
    #if progress:
    #    iterator = tqdm(iterator, desc="Computing Chi² kernel (sparse)", unit="batch")

    # Prepara Y in formato CSC per accesso più efficiente alle colonne comuni
    Y_csc = Y.tocsc()

    for start in iterator:
        end = min(start + batch_size, n_samples_X)
        X_batch = X[start:end]

        # Output batch
        K_chunk = np.zeros((X_batch.shape[0], n_samples_Y), dtype=np.float32)

        # Itera sulle righe del batch
        for i, (x_indptr_start, x_indptr_end) in enumerate(zip(X_batch.indptr[:-1], X_batch.indptr[1:])):
            x_idx = X_batch.indices[x_indptr_start:x_indptr_end]
            x_data = X_batch.data[x_indptr_start:x_indptr_end]

            # Per ogni colonna j (cioè riga di Y)
            for j in range(n_samples_Y):
                y_col = Y_csc.getcol(j)
                if y_col.nnz == 0:
                    continue

                # Indici e dati comuni
                y_idx = y_col.indices
                y_data = y_col.data

                # Intersezione degli indici non-zero
                common = np.intersect1d(x_idx, y_idx, assume_unique=True)
                if common.size == 0:
                    continue

                x_common = x_data[np.searchsorted(x_idx, common)]
                y_common = y_data[np.searchsorted(y_idx, common)]

                # Formula Chi²: 2 * x * y / (x + y)
                num = 2 * x_common * y_common
                denom = x_common + y_common + 1e-12
                val = np.sum(num / denom)

                K_chunk[i, j] = np.exp(-gamma * val)

        yield K_chunk"""
