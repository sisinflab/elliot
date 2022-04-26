"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import math

import numpy as np
import random
import os
from tqdm import tqdm
import tensorflow as tf


from .ConvMFCNN import CNN_module


class convMF(object):

    def __init__(self,
                 data,
                 lambda_u,
                 lambda_i,
                 embedding_size,
                 factors_dim,
                 kernel_per_ws,
                 drop_out_rate,
                 epochs,
                 vocab_size,
                 max_len,
                 CNN_X,
                 init_W,
                 batch_size,
                 give_item_weight,
                 random_seed,
                 name="ConvMF",
                 **kwargs):

        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        self.a = 1
        self.b = 0
        self._data = data
        self.seed = random_seed
        self.lambda_u = lambda_u
        self.lambda_i = lambda_i
        self.kernel_per_ws = kernel_per_ws
        self.epochs = epochs
        self.embedding_size = embedding_size
        self.factors_dim = factors_dim
        self.data_ratings = self._data.sp_i_train_ratings
        self.data_ratings_transpose = self._data.sp_i_train_ratings.T
        self.dropout_rate = drop_out_rate
        self.max_len = max_len
        self.init_W = init_W
        self.CNN_X = CNN_X
        self.batch_size = batch_size


        self.user_total = self.data_ratings.shape[0]
        self.item_total = self.data_ratings.shape[1]

        if give_item_weight:
            self.item_weight = np.array([math.sqrt(len(r))
                                    for r in self.data_ratings_transpose.tolil().rows], dtype=float)
            self.item_weight = (float(self.item_total) / self.item_weight.sum()) * self.item_weight
        else:
            self.item_weight = np.ones(self.item_total, dtype=float)

        self.user_embeddings = np.random.uniform(size=(self.user_total, self.factors_dim))

        self.cnn_module = CNN_module(self.factors_dim, vocab_size, self.dropout_rate, self.batch_size,
                                     self.embedding_size, self.max_len, self.kernel_per_ws, self.init_W, self.seed)
        self.theta = self.cnn_module.get_projection_layer(self.CNN_X)
        self.item_embeddings = self.theta
        self.pred_mat = None
        self.lambda_u_matrix = self.lambda_u * np.eye(self.factors_dim, dtype=np.float32)
        self.lambda_i_matrix = self.lambda_i * np.eye(self.factors_dim, dtype=np.float32)

    def get_config(self):
        raise NotImplementedError

    def train_step(self, **kwargs):
        with tqdm(total=self.user_total, position=0, leave=True) as pbar:
            loss = 0
            pbar.set_description("Users")
            VV = self.b * (self.item_embeddings.T.dot(self.item_embeddings)) + self.lambda_u_matrix
            sub_loss = np.zeros(self.user_total)
            for u, idx_item in enumerate(self.data_ratings.tolil().rows):
                # idx_item = self.data_ratings[u].indices
                V_i = self.item_embeddings[idx_item]
                R_i = self.data_ratings[u].data
                A = VV + (self.a - self.b) * (V_i.T.dot(V_i))
                B = (self.a * V_i * (np.tile(R_i, (self.factors_dim, 1)).T)).sum(0)

                self.user_embeddings[u] = np.linalg.solve(A, B)

                sub_loss[u] = -0.5 * self.lambda_u * np.dot(self.user_embeddings[u], self.user_embeddings[u])
                pbar.update()

            loss = loss + np.sum(sub_loss)
            pbar.reset(total=self.item_total)
            pbar.set_description("Items")
            UU = self.b * self.user_embeddings.T.dot(self.user_embeddings)
            sub_loss = np.zeros(self.item_total)
            for i, idx_user in enumerate(self.data_ratings_transpose.tolil().rows):
                # idx_user = self.data_ratings.T[i].indices
                U_j = self.user_embeddings[idx_user]
                R_j = self.data_ratings_transpose[i].data

                tmp_A = UU + (self.a - self.b) * (U_j.T.dot(U_j))
                A = tmp_A + self.lambda_i_matrix * self.item_weight[i]
                B = (self.a * U_j * (np.tile(R_j, (self.factors_dim, 1)).T)).sum(0) \
                    + self.lambda_i * self.item_weight[i] * self.theta[i]
                self.item_embeddings[i] = np.linalg.solve(A, B)

                sub_loss[i] = -0.5 * np.square(R_j * self.a).sum()
                sub_loss[i] = sub_loss[i] + self.a * np.sum((U_j.dot(self.item_embeddings[i])) * R_j)
                sub_loss[i] = sub_loss[i] - 0.5 * np.dot(self.item_embeddings[i].dot(tmp_A), self.item_embeddings[i])
                pbar.update()

            loss = loss + np.sum(sub_loss)
            history = self.cnn_module.train(self.CNN_X, self.item_embeddings, self.item_weight, self.seed)
            self.theta = self.cnn_module.get_projection_layer(self.CNN_X)
            cnn_loss = history.history['loss'][-1]

            loss = loss - 0.5 * self.lambda_i * cnn_loss * self.item_total

        return loss

    def prepare_predictions(self):
        self.pred_mat = self.user_embeddings.dot(self.item_embeddings.T)

    def predict(self, user, item):
        return self.pred_mat[self._data.public_users[user], self._data.public_items[item]]

    def get_all_topks(self, mask, k, user_map, item_map):
        masking = np.where(mask, self.pred_mat, -np.inf)
        partial_index = np.argpartition(masking, -k, axis=1)[:, -k:]
        masking_partition = np.take_along_axis(masking, partial_index, axis=1)
        masking_partition_index = masking_partition.argsort(axis=1)[:, ::-1]
        partial_index = np.take_along_axis(partial_index, masking_partition_index, axis=1)
        masking_partition = np.take_along_axis(masking_partition, masking_partition_index, axis=1)
        predictions_top_k = {
            user_map[u]: list(map(lambda x: (item_map.get(x[0]), x[1]), zip(*map(lambda x: x, top)))) for
            u, top in enumerate(zip(*(partial_index.tolist(), masking_partition.tolist())))}
        return predictions_top_k