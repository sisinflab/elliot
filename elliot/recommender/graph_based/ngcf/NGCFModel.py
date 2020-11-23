"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import logging
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

tf.random.set_seed(42)

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class NGCFModel(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

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
                 plain_adj,
                 norm_adj,
                 mean_adj,
                 name="NGFC",
                 **kwargs
                 ):
        super().__init__(name=name, **kwargs)

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
        self.plain_adj = plain_adj
        self.norm_adj = norm_adj
        self.mean_adj = mean_adj

        # Generate a set of adjacency sub-matrix.
        if len(self.node_dropout):
            # node dropout.
            # self.A_fold_hat = self._split_A_hat_node_dropout()
            print(1)
        else:
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

        # Gu and Gi are different from the other recommenders, because in this case they are obtained as:
        # Gu = Gu_0 || Gu_1 || ... || Gu_L
        # Gi = Gi_0 || Gi_1 || ... || Gi_L
        self.Gu = tf.Variable(self.initializer([self.num_users, self.embed_k * (self.n_layers + 1)]), name='Gu')
        self.Gi = tf.Variable(self.initializer([self.num_items, self.embed_k * (self.n_layers + 1)]), name='Gi')

        self.Graph = dict()

        self.weight_size_list = [self.embed_k] + self.weight_size

        for k in range(self.n_layers):
            self.Graph['W_gc_%d' % k] = tf.Variable(
                self.initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            self.Graph['b_gc_%d' % k] = tf.Variable(
                self.initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            self.Graph['W_bi_%d' % k] = tf.Variable(
                self.initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            self.Graph['b_bi_%d' % k] = tf.Variable(
                self.initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

    def _propagate_embeddings(self):
        # extract gu_0 and gi_0 to begin embedding updating for L layers
        gu_0 = self.Gu[:, :self.embed_k]
        gi_0 = self.Gi[:, :self.embed_k]

        ego_embeddings = tf.concat([gu_0, gi_0], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse.sparse_dense_matmul(self.A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.Graph['W_gc_%d' % k]) + self.Graph['b_gc_%d' % k])

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.Graph['W_bi_%d' % k]) + self.Graph['b_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.message_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        gu, gi = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        self.Gu.assign(gu)
        self.Gi.assign(gi)

    def _split_A_hat(self):
        A_fold_hat = []

        fold_len = (self.num_users + self.num_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.num_users + self.num_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(self.norm_adj[start:end]))
        return A_fold_hat

    # def _split_A_hat_node_dropout(self):
    #     A_fold_hat = []
    #
    #     fold_len = (self.num_users + self.num_items) // self.n_fold
    #     for i_fold in range(self.n_fold):
    #         start = i_fold * fold_len
    #         if i_fold == self.n_fold - 1:
    #             end = self.num_users + self.num_items
    #         else:
    #             end = (i_fold + 1) * fold_len
    #
    #         # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
    #         temp = self._convert_sp_mat_to_sp_tensor(self.norm_adj[start:end])
    #         n_nonzero_temp = self.norm_adj[start:end].count_nonzero()
    #         A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))
    #
    #     return A_fold_hat

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

    # def predict_all(self):
    #     """
    #     Get full predictions on the whole users/items matrix.
    #
    #     Returns:
    #         The matrix of predicted values.
    #     """
    #     return tf.matmul(self.Gu, self.Gi, transpose_b=True)

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
                                                [tf.nn.l2_loss(value) for _, value in self.Graph.items()])

            # Loss to be optimized
            loss += reg_loss

        grads = tape.gradient(loss, [self.Gu, self.Gi] +
                              [value for _, value in self.Graph.items()])
        self.optimizer.apply_gradients(zip(grads, [self.Gu, self.Gi] +
                                           [value for _, value in self.Graph.items()]))

        return loss

    def get_config(self):
        raise NotImplementedError

    # @tf.function
    # def call(self, inputs, training=None, **kwargs):
    #     z_mean = self.encoder(inputs, training=training)
    #     reconstructed = self.decoder(z_mean)
    #     return reconstructed

    # @tf.function
    # def train_step(self, batch):
    #     with tf.GradientTape() as tape:
    #         # Clean Inference
    #         logits = self.call(inputs=batch, training=True)
    #         log_softmax_var = tf.nn.log_softmax(logits)
    #
    #         # per-user average negative log-likelihood
    #         loss = -tf.reduce_mean(tf.reduce_sum(
    #             log_softmax_var * batch, axis=1))
    #
    #     grads = tape.gradient(loss, self.trainable_weights)
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    #
    #     return loss

    # @tf.function
    # def predict(self, inputs, training=False, **kwargs):
    #     """
    #     Get full predictions on the whole users/items matrix.
    #
    #     Returns:
    #         The matrix of predicted values.
    #     """
    #
    #     logits = self.call(inputs=inputs, training=training)
    #     log_softmax_var = tf.nn.log_softmax(logits)
    #     return log_softmax_var

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
