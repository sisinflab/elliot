"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class LightGCNModel(keras.Model):

    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 n_layers,
                 n_fold,
                 adjacency,
                 laplacian,
                 random_seed,
                 name="LightGCN",
                 **kwargs
                 ):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.n_fold = n_fold
        self.n_layers = n_layers
        self.adjacency = adjacency
        self.laplacian = laplacian

        # Generate a set of adjacency sub-matrix.
        self.A_fold_hat = self._split_A_hat()

        self.initializer = tf.initializers.GlorotUniform()
        # Initialize Model Parameters
        self._create_weights()

        self.optimizer = tf.optimizers.Adam(self.learning_rate)

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _create_weights(self):
        self.Gu = tf.Variable(tf.zeros([self.num_users, self.embed_k]), name='Gu')
        self.Gi = tf.Variable(tf.zeros([self.num_items, self.embed_k]), name='Gi')

    @tf.function
    def _propagate_embeddings(self):
        gu_0 = self.Gu
        gi_0 = self.Gi
        ego_embeddings = tf.concat([gu_0, gi_0], axis=0)
        all_embeddings = [ego_embeddings]
        all_alphas = [1]

        for k in range(1, self.n_layers + 1):
            # This matrix multiplication is performed in smaller folders of the adj matrix to fit into memory
            laplacian_embeddings = []
            for f in range(self.n_fold):
                laplacian_embeddings.append(tf.sparse.sparse_dense_matmul(self.A_fold_hat[f], ego_embeddings))

            laplacian_embeddings = tf.concat(laplacian_embeddings, 0)
            ego_embeddings = laplacian_embeddings

            all_embeddings += [laplacian_embeddings]

            all_alphas += [1 / (1 + k)]

        all_embeddings = [emb * alpha for alpha, emb in zip(all_alphas, all_embeddings)]
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        gu, gi = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        self.Gu.assign(gu)
        self.Gi.assign(gi)

    @tf.function
    def _split_A_hat(self):
        A_fold_hat = []

        fold_len = (self.num_users + self.num_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.num_users + self.num_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(self.laplacian[start:end]))

        return A_fold_hat

    @tf.function
    def call(self, inputs, **kwargs):
        """
        Generates prediction for passed users and items indices

        Args:
            inputs: user, item (batch)
            the `Network` in training mode or inference mode.

        Returns:
            prediction and extracted model parameters
        """
        user, item = inputs
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))

        xui = tf.reduce_sum(gamma_u * gamma_i, 1)

        return xui, gamma_u, gamma_i

    @tf.function
    def predict(self, start, stop, **kwargs):
        return tf.matmul(self.Gu[start:stop], self.Gi, transpose_b=True)

    @tf.function
    def train_step(self, batch):
        """
        Apply a single training step on one batch.

        Args:
            batch: batch used for the current train step

        Returns:
            loss value at the current batch
        """
        user, pos, neg = batch
        with tf.GradientTape() as tape:
            # Clean Inference
            self._propagate_embeddings()
            xu_pos, gamma_u, gamma_pos = self(inputs=(user, pos), training=True)
            xu_neg, _, gamma_neg = self(inputs=(user, neg), training=True)

            difference = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-difference))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg)]) * 2

            # Loss to be optimized
            loss += reg_loss

        grads = tape.gradient(loss, [self.Gu, self.Gi])
        self.optimizer.apply_gradients(zip(grads, [self.Gu, self.Gi]))

        return loss

    def get_config(self):
        raise NotImplementedError

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
