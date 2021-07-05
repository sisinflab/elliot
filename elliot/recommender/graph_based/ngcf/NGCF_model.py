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


class NGCFModel(keras.Model):

    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 weight_size,
                 n_layers,
                 node_dropout,
                 message_dropout,
                 n_fold,
                 adjacency,
                 laplacian,
                 random_seed,
                 name="NGFC",
                 **kwargs
                 ):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.weight_size = weight_size
        self.n_layers = n_layers
        self.node_dropout = node_dropout
        self.message_dropout = message_dropout
        self.n_fold = n_fold
        self.adjacency = adjacency
        self.laplacian = laplacian

        # Generate a set of adjacency sub-matrix.
        if len(self.node_dropout):
            # node dropout.
            self.A_fold_hat = self._split_A_hat(dropout=True)
        else:
            self.A_fold_hat = self._split_A_hat(dropout=False)

        self.initializer = tf.initializers.GlorotUniform()
        # Initialize Model Parameters
        self._create_weights()

        self.optimizer = tf.optimizers.Adam(self.learning_rate)

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    @staticmethod
    def _dropout_sparse(X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random.uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse.retain(X, dropout_mask)

        return pre_out * tf.math.divide(1., keep_prob)

    def _create_weights(self):
        # Gu and Gi are obtained as:
        # Gu = Gu_0 || Gu_1 || ... || Gu_L
        # Gi = Gi_0 || Gi_1 || ... || Gi_L
        self.weight_size_list = [self.embed_k] + self.weight_size
        self.Gu = tf.Variable(tf.zeros([self.num_users, sum(self.weight_size_list)]), name='Gu')
        self.Gi = tf.Variable(tf.zeros([self.num_items, sum(self.weight_size_list)]), name='Gi')

        self.GraphLayers = dict()

        for k in range(self.n_layers):
            self.GraphLayers['W_1_%d' % k] = tf.Variable(
                self.initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_1_%d' % k)
            self.GraphLayers['b_1_%d' % k] = tf.Variable(
                self.initializer([1, self.weight_size_list[k + 1]]), name='b_1_%d' % k)

            self.GraphLayers['W_2_%d' % k] = tf.Variable(
                self.initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_2_%d' % k)
            self.GraphLayers['b_2_%d' % k] = tf.Variable(
                self.initializer([1, self.weight_size_list[k + 1]]), name='b_2_%d' % k)

    @tf.function
    def _propagate_embeddings(self):
        # Extract gu_0 and gi_0 to begin embedding updating for L layers
        gu_0 = self.Gu[:, :self.embed_k]
        gi_0 = self.Gi[:, :self.embed_k]

        ego_embeddings = tf.concat([gu_0, gi_0], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            # This matrix multiplication is performed in smaller folders of the adj matrix to fit into memory
            laplacian_embeddings = []
            for f in range(self.n_fold):
                laplacian_embeddings.append(tf.sparse.sparse_dense_matmul(self.A_fold_hat[f], ego_embeddings))
            laplacian_embeddings = tf.concat(laplacian_embeddings, 0)

            first_contribution = tf.matmul(
                    laplacian_embeddings + ego_embeddings,
                    self.GraphLayers['W_1_%d' % k]
                ) + self.GraphLayers['b_1_%d' % k]

            second_contribution = tf.multiply(ego_embeddings, laplacian_embeddings)
            second_contribution = tf.matmul(
                    second_contribution,
                    self.GraphLayers['W_2_%d' % k]
                ) + self.GraphLayers['b_2_%d' % k]

            ego_embeddings = tf.nn.leaky_relu(first_contribution + second_contribution)

            ego_embeddings = tf.nn.dropout(ego_embeddings, self.message_dropout[k])

            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        gu, gi = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        self.Gu.assign(gu)
        self.Gi.assign(gi)

    @tf.function
    def _split_A_hat(self, dropout=False):
        A_fold_hat = []

        fold_len = (self.num_users + self.num_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.num_users + self.num_items
            else:
                end = (i_fold + 1) * fold_len

            if not dropout:
                A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(self.laplacian[start:end]))
            else:
                temp = self._convert_sp_mat_to_sp_tensor(self.laplacian[start:end])
                n_nonzero_temp = self.laplacian[start:end].count_nonzero()
                A_fold_hat.append(self._dropout_sparse(temp, self.node_dropout[0], n_nonzero_temp))

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
                                                 tf.nn.l2_loss(gamma_neg)] +
                                                [tf.nn.l2_loss(value) for _, value in self.GraphLayers.items()]) * 2

            # Loss to be optimized
            loss += reg_loss

        grads = tape.gradient(loss, [self.Gu, self.Gi] +
                              [value for _, value in self.GraphLayers.items()])
        self.optimizer.apply_gradients(zip(grads, [self.Gu, self.Gi] +
                                           [value for _, value in self.GraphLayers.items()]))

        return loss

    def get_config(self):
        raise NotImplementedError

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
