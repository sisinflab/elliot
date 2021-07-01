#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17
@author: Maurizio Ferrari Dacrema

"""
import pickle

import numpy as np
import time, sys
import scipy.sparse as sp


def check_matrix(X, format='csc', dtype=np.float32):
    """
    This function takes a matrix as input and transforms it into the specified format.
    The matrix in input can be either sparse or ndarray.
    If the matrix in input has already the desired format, it is returned as-is
    the dtype parameter is always applied and the default is np.float32
    :param X:
    :param format:
    :param dtype:
    :return:
    """

    if format == 'csc' and not isinstance(X, sp.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sp.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sp.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sp.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sp.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sp.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sp.lil_matrix):
        return X.tolil().astype(dtype)
    elif isinstance(X, np.ndarray):
        X = sp.csr_matrix(X, dtype=dtype)
        X.eliminate_zeros()
        return check_matrix(X, format=format, dtype=dtype)
    else:
        return X.astype(dtype)


class AiolliSimilarity(object):
    def __init__(self, data,
                 maxk=40,
                 shrink=100,
                 similarity='cosine',
                 implicit=False,
                 normalize=True,
                 asymmetric_alpha=0.5,
                 tversky_alpha = 1.0,
                 tversky_beta = 1.0,
                 row_weights = None):
        """
        ItemKNN recommender
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        maxk : int, the max similar items number
        shrink : float, shrink similarity value
        similarity : str, way to calculate similarity
        normalize : bool, whether calculate similarity with normalized value
        """
        self._data = data
        self._implicit = implicit

        if self._implicit:
            self._train_set = data.sp_i_train
        else:
            self._train_set = self._data.sp_i_train_ratings

        self._private_users = self._data.private_users
        self._public_users = self._data.public_users
        self._private_items = self._data.private_items
        self._public_items = self._data.public_items

        self.user_num = self._data.num_users
        self.item_num = self._data.num_items

        self.k = maxk
        self.shrink = shrink
        self.normalize = normalize
        self.similarity = similarity
        self.asymmetric_alpha = asymmetric_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.row_weights = row_weights

        self.RECOMMENDER_NAME = "UserKNNCFRecommender"

        # self.pred_mat = None
        # self.yr = None

    def initialize(self):
        # self.yr = defaultdict(list)
        # for _, row in self._train_set.iterrows():
        #     self.yr[int(row['user'])].append((int(row['item']), row['rating']))

        # train = self._convert_df(self.user_num, self.item_num, self._train_set)
        train = self._train_set.tocsc()

        cold_user_mask = np.ediff1d(train.tocsc().indptr) == 0

        if cold_user_mask.any():
            print("{}: Detected {} ({:.2f} %) cold items.".format(
                self.RECOMMENDER_NAME, cold_user_mask.sum(), cold_user_mask.sum() / len(cold_user_mask) * 100))

        similarity = Compute_Similarity(train.T,
                                        shrink=self.shrink,
                                        topK=self.k,
                                        normalize=self.normalize,
                                        similarity=self.similarity,
                                        asymmetric_alpha=self.asymmetric_alpha,
                                        tversky_alpha=self.tversky_alpha,
                                        tversky_beta=self.tversky_beta,
                                        row_weights=self.row_weights)

        w_sparse = similarity.compute_similarity()
        w_sparse = w_sparse.tocsc()

        # self.pred_mat = w_sparse.dot(train).tolil()
        self.pred_mat = w_sparse.dot(train).toarray()

    def get_user_recs(self, u, mask, k):
        user_id = self._data.public_users.get(u)
        user_recs = self.pred_mat[user_id]
        # user_items = self._ratings[u].keys()
        user_recs_mask = mask[user_id]
        user_recs[~user_recs_mask] = -np.inf
        indices, values = zip(*[(self._data.private_items.get(u_list[0]), u_list[1])
                              for u_list in enumerate(user_recs)])

        # indices, values = zip(*predictions.items())
        indices = np.array(indices)
        values = np.array(values)
        local_k = min(k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]

    # def get_user_recs(self, user, k=100):
    #     user_items = self._data.train_dict[user].keys()
    #     predictions = {i: self.predict(user, i) for i in self._data.items if i not in user_items}
    #     indices, values = zip(*predictions.items())
    #     indices = np.array(indices)
    #     values = np.array(values)
    #     local_k = min(k, len(values))
    #     partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
    #     real_values = values[partially_ordered_preds_indices]
    #     real_indices = indices[partially_ordered_preds_indices]
    #     local_top_k = real_values.argsort()[::-1]
    #     return [(real_indices[item], real_values[item]) for item in local_top_k]

    # def predict(self, u, i):
    #     indexed_user = self._public_users[u]
    #     indexed_item = self._public_items[i]
    #     if indexed_user >= self.user_num or indexed_item >= self.item_num:
    #         raise ValueError('User and/or item is unkown.')
    #
    #     return self.pred_mat[indexed_user, indexed_item]

    def _convert_df(self, user_num, item_num, df):
        """Process Data to make ItemKNN available"""
        ratings = list(df['rating'])
        rows = list(df['user'])
        cols = list(df['item'])

        mat = sp.csc_matrix((ratings, (rows, cols)), shape=(user_num, item_num))

        return mat

    def get_model_state(self):
        saving_dict = {}
        saving_dict['_preds'] = self._preds
        saving_dict['_similarity'] = self._similarity
        saving_dict['_num_neighbors'] = self._num_neighbors
        saving_dict['_implicit'] = self._implicit
        return saving_dict

    def set_model_state(self, saving_dict):
        self._preds = saving_dict['_preds']
        self._similarity = saving_dict['_similarity']
        self._num_neighbors = saving_dict['_num_neighbors']
        self._implicit = saving_dict['_implicit']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)


class Compute_Similarity:

    def __init__(self, dataMatrix, topK=100, shrink=0, normalize=True,
                 asymmetric_alpha=0.5, tversky_alpha=1.0, tversky_beta=1.0,
                 similarity="cosine", row_weights=None):
        """
        Computes the cosine similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param shrink:
        :param normalize:           If True divide the dot product by the product of the norms
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param asymmetric_alpha     Coefficient alpha for the asymmetric cosine
        :param similarity:  "cosine"        computes Cosine similarity
                            "adjusted"      computes Adjusted Cosine, removing the average of the users
                            "asymmetric"    computes Asymmetric Cosine
                            "pearson"       computes Pearson Correlation, removing the average of the items
                            "jaccard"       computes Jaccard similarity for binary interactions using Tanimoto
                            "dice"          computes Dice similarity for binary interactions
                            "tversky"       computes Tversky similarity for binary interactions
                            "tanimoto"      computes Tanimoto coefficient for binary interactions
        """
        """
        Asymmetric Cosine as described in: 
        Aiolli, F. (2013, October). Efficient top-n recommendation for very large scale binary rated datasets. In Proceedings of the 7th ACM conference on Recommender systems (pp. 273-280). ACM.

        """

        # super(Compute_Similarity_Python, self).__init__()

        self.shrink = shrink
        self.normalize = normalize

        self.n_rows, self.n_columns = dataMatrix.shape
        self.TopK = min(topK, self.n_columns)

        self.asymmetric_alpha = asymmetric_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        self.dataMatrix = dataMatrix.copy()

        self.adjusted_cosine = False
        self.asymmetric_cosine = False
        self.pearson_correlation = False
        self.tanimoto_coefficient = False
        self.dice_coefficient = False
        self.tversky_coefficient = False

        if similarity == "adjusted":
            self.adjusted_cosine = True
        elif similarity == "asymmetric":
            self.asymmetric_cosine = True
        elif similarity == "pearson":
            self.pearson_correlation = True
        elif similarity == "jaccard" or similarity == "tanimoto":
            self.tanimoto_coefficient = True
            # Tanimoto has a specific kind of normalization
            self.normalize = False

        elif similarity == "dice":
            self.dice_coefficient = True
            self.normalize = False

        elif similarity == "tversky":
            self.tversky_coefficient = True
            self.normalize = False

        elif similarity == "cosine":
            pass
        else:
            raise ValueError("Cosine_Similarity: value for parameter 'mode' not recognized."
                             " Allowed values are: 'cosine', 'pearson', 'adjusted', 'asymmetric', 'jaccard', 'tanimoto',"
                             "dice, tversky."
                             " Passed value was '{}'".format(similarity))

        self.use_row_weights = False

        if row_weights is not None:

            if dataMatrix.shape[0] != len(row_weights):
                raise ValueError("Cosine_Similarity: provided row_weights and dataMatrix have different number of rows."
                                 "Col_weights has {} columns, dataMatrix has {}.".format(len(row_weights),
                                                                                         dataMatrix.shape[0]))

            self.use_row_weights = True
            self.row_weights = row_weights.copy()
            self.row_weights_diag = sp.diags(self.row_weights)

            self.dataMatrix_weighted = self.dataMatrix.T.dot(self.row_weights_diag).T

    def applyAdjustedCosine(self):
        """
        Remove from every data point the average for the corresponding row
        :return:
        """

        self.dataMatrix = check_matrix(self.dataMatrix, 'csr')

        interactionsPerRow = np.diff(self.dataMatrix.indptr)

        nonzeroRows = interactionsPerRow > 0
        sumPerRow = np.asarray(self.dataMatrix.sum(axis=1)).ravel()

        rowAverage = np.zeros_like(sumPerRow)
        rowAverage[nonzeroRows] = sumPerRow[nonzeroRows] / interactionsPerRow[nonzeroRows]

        # Split in blocks to avoid duplicating the whole data structure
        start_row = 0
        end_row = 0

        blockSize = 1000

        while end_row < self.n_rows:
            end_row = min(self.n_rows, end_row + blockSize)

            self.dataMatrix.data[self.dataMatrix.indptr[start_row]:self.dataMatrix.indptr[end_row]] -= \
                np.repeat(rowAverage[start_row:end_row], interactionsPerRow[start_row:end_row])

            start_row += blockSize

    def applyPearsonCorrelation(self):
        """
        Remove from every data point the average for the corresponding column
        :return:
        """

        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        interactionsPerCol = np.diff(self.dataMatrix.indptr)

        nonzeroCols = interactionsPerCol > 0
        sumPerCol = np.asarray(self.dataMatrix.sum(axis=0)).ravel()

        colAverage = np.zeros_like(sumPerCol)
        colAverage[nonzeroCols] = sumPerCol[nonzeroCols] / interactionsPerCol[nonzeroCols]

        # Split in blocks to avoid duplicating the whole data structure
        start_col = 0
        end_col = 0

        blockSize = 1000

        while end_col < self.n_columns:
            end_col = min(self.n_columns, end_col + blockSize)

            self.dataMatrix.data[self.dataMatrix.indptr[start_col]:self.dataMatrix.indptr[end_col]] -= \
                np.repeat(colAverage[start_col:end_col], interactionsPerCol[start_col:end_col])

            start_col += blockSize

    def useOnlyBooleanInteractions(self):

        # Split in blocks to avoid duplicating the whole data structure
        start_pos = 0
        end_pos = 0

        blockSize = 1000

        while end_pos < len(self.dataMatrix.data):
            end_pos = min(len(self.dataMatrix.data), end_pos + blockSize)

            self.dataMatrix.data[start_pos:end_pos] = np.ones(end_pos - start_pos)

            start_pos += blockSize

    def compute_similarity(self, start_col=None, end_col=None, block_size=100):
        """
        Compute the similarity for the given dataset
        :param self:
        :param start_col: column to begin with
        :param end_col: column to stop before, end_col is excluded
        :return:
        """

        values = []
        rows = []
        cols = []

        start_time = time.time()
        start_time_print_batch = start_time
        processedItems = 0

        if self.adjusted_cosine:
            self.applyAdjustedCosine()

        elif self.pearson_correlation:
            self.applyPearsonCorrelation()

        elif self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient:
            self.useOnlyBooleanInteractions()

        # We explore the matrix column-wise
        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        # Compute sum of squared values to be used in normalization
        sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()

        # Tanimoto does not require the square root to be applied
        if not (self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient):
            sumOfSquared = np.sqrt(sumOfSquared)

        if self.asymmetric_cosine:
            sumOfSquared_to_1_minus_alpha = np.power(sumOfSquared, 2 * (1 - self.asymmetric_alpha))
            sumOfSquared_to_alpha = np.power(sumOfSquared, 2 * self.asymmetric_alpha)

        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        start_col_local = 0
        end_col_local = self.n_columns

        if start_col is not None and start_col > 0 and start_col < self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col > start_col_local and end_col < self.n_columns:
            end_col_local = end_col

        start_col_block = start_col_local

        this_block_size = 0

        # Compute all similarities for each item using vectorization
        while start_col_block < end_col_local:

            end_col_block = min(start_col_block + block_size, end_col_local)
            this_block_size = end_col_block - start_col_block

            # All data points for a given item
            item_data = self.dataMatrix[:, start_col_block:end_col_block]
            item_data = item_data.toarray().squeeze()

            # If only 1 feature avoid last dimension to disappear
            if item_data.ndim == 1:
                item_data = np.atleast_2d(item_data)

            if self.use_row_weights:
                this_block_weights = self.dataMatrix_weighted.T.dot(item_data)

            else:
                # Compute item similarities
                this_block_weights = self.dataMatrix.T.dot(item_data)

            for col_index_in_block in range(this_block_size):

                if this_block_size == 1:
                    this_column_weights = this_block_weights
                else:
                    this_column_weights = this_block_weights[:, col_index_in_block]

                columnIndex = col_index_in_block + start_col_block
                this_column_weights[columnIndex] = 0.0

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize:

                    if self.asymmetric_cosine:
                        denominator = sumOfSquared_to_alpha[
                                          columnIndex] * sumOfSquared_to_1_minus_alpha + self.shrink + 1e-6
                    else:
                        denominator = sumOfSquared[columnIndex] * sumOfSquared + self.shrink + 1e-6

                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                # Apply the specific denominator for Tanimoto
                elif self.tanimoto_coefficient:
                    denominator = sumOfSquared[columnIndex] + sumOfSquared - this_column_weights + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.dice_coefficient:
                    denominator = sumOfSquared[columnIndex] + sumOfSquared + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.tversky_coefficient:
                    denominator = this_column_weights + \
                                  (sumOfSquared[columnIndex] - this_column_weights) * self.tversky_alpha + \
                                  (sumOfSquared - this_column_weights) * self.tversky_beta + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                # If no normalization or tanimoto is selected, apply only shrink
                elif self.shrink != 0:
                    this_column_weights = this_column_weights / self.shrink

                # this_column_weights = this_column_weights.toarray().ravel()

                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                relevant_items_partition = (-this_column_weights).argpartition(self.TopK - 1)[0:self.TopK]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_column_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values.extend(this_column_weights[top_k_idx][notZerosMask])
                rows.extend(top_k_idx[notZerosMask])
                cols.extend(np.ones(numNotZeros) * columnIndex)

            # Add previous block size
            processedItems += this_block_size

            if time.time() - start_time_print_batch >= 30 or end_col_block == end_col_local:
                columnPerSec = processedItems / (time.time() - start_time + 1e-9)

                print("Similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                    processedItems, processedItems / (end_col_local - start_col_local) * 100, columnPerSec,
                                    (time.time() - start_time) / 60))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()

            start_col_block += block_size

        # End while on columns

        W_sparse = sp.csr_matrix((values, (rows, cols)),
                                 shape=(self.n_columns, self.n_columns),
                                 dtype=np.float32)

        return W_sparse
