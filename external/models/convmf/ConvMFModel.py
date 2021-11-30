"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .ConvMFCNN import CNN_module


class convMF(keras.Model):

    def __init__(self,
                 lambda_u,
                 lambda_i,
                 embedding_size,
                 factors_dim,
                 kernel_per_ws,
                 drop_out_rate,
                 epochs,
                 data_ratings,
                 vocab_size,
                 max_len,
                 CNN_X,
                 random_seed,
                 name="cofm",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self.a = 1
        self.b = 0

        self.lambda_u = lambda_u
        self.lambda_i = lambda_i
        self.kernel_per_ws = kernel_per_ws
        self.epochs = epochs
        self.embedding_size = embedding_size
        self.factors_dim = factors_dim
        self.data_ratings = data_ratings
        self.dropout_rate = drop_out_rate
        self.max_len = max_len
        self.init_W = None

        self.user_total = self.data_ratings.shape[0]
        self.item_total = self.data_ratings.shape[1]

        self.item_weight = np.ones(self.item_total, dtype=float)

        initializer = keras.initializers.GlorotNormal()

        self.user_embeddings = tf.Variable(initializer(shape=(self.user_total, self.embedding_size)),
                                           shape=[self.user_total, self.embedding_size],
                                           trainable=False)

        self.cnn_module = CNN_module(self.factors_dim, vocab_size, self.dropout_rate,
                                     self.embedding_size, self.max_len, self.kernel_per_ws, self.init_W)
        self.theta = self.cnn_module.get_projection_layer(CNN_X)


        def get_config(self):
            raise NotImplementedError

    # @tf.function
    def call(self, inputs, training=None, **kwargs):

        u_ids, i_ids = inputs
        batch_size = len(u_ids)

        u_e = self.user_embeddings(tf.squeeze(u_ids))
        i_e = self.item_embeddings(tf.squeeze(i_ids))
        u_b = self.user_bias(tf.squeeze(u_ids))
        i_b = self.item_bias(tf.squeeze(i_ids))

        score = self.bias + u_b + i_b + tf.squeeze(tf.matmul(tf.expand_dims(u_e, len(tf.shape(u_e))-1),
                                                             tf.expand_dims(i_e, len(tf.shape(i_e)))), axis=-1)

        return score

    @tf.function
    def train_step(self, batch, **kwargs):
        loss = 0

        VV = self.b * (self.item_embeddings.T.dot(self.item_embeddings)) + self.lambda_u * np.eye(self.dimension)
        sub_loss = np.zeros(self.user_total)
        for u in range(self.user_total):
            idx_item = train_user[0][u]
            V_i = self.item_embeddings[idx_item]
            R_i = self.data_ratings[u]
            A = VV + (self.a - self.b) * (V_i.T.dot(V_i))
            B = (self.a * V_i * (np.tile(R_i, (self.dimension, 1)).T)).sum(0)

            self.user_embeddings[u] = tf.linalg.solve(A, B)

            sub_loss[u] = -0.5 * self.lambda_u * np.dot(self.user_embeddings[u], self.user_embeddings[u])

        loss = loss + np.sum(sub_loss)

        UU = self.b * (self.user_embeddings.T.dot(self.user_embeddings))
        sub_loss = np.zeros(self.item_total)
        for i in range(self.item_total):
            idx_user = train_item[0][i]
            U_j = self.user_embeddings[idx_user]
            R_j = self.data_ratings[j]

            tmp_A = UU + (self.a - self.b) * (U_j.T.dot(U_j))
            A = tmp_A + self.lambda_v * self.item_weight[i] * np.eye(self.dimension)
            B = (self.a * U_j * (np.tile(R_j, (self.dimension, 1)).T)
                 ).sum(0) + self.lambda_v * self.item_weight[i] * self.theta[i]
            self.item_embeddings[i] = tf.linalg.solve(A, B)

            sub_loss[i] = -0.5 * np.square(R_j * a).sum()
            sub_loss[i] = sub_loss[i] + self.a * np.sum((U_j.dot(self.item_embeddings[i])) * R_j)
            sub_loss[i] = sub_loss[i] - 0.5 * np.dot(self.item_embeddings[i].dot(tmp_A), self.item_embeddings[i])

        loss = loss + np.sum(sub_loss)

        return loss

    @tf.function
    def predict(self, inputs, training=False, **kwargs):
        score = self.call(inputs=inputs, training=training, is_rec=True)
        return score

    @tf.function
    def get_recs(self, inputs, training=False, **kwargs):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        u_ids, i_ids = inputs

        score = self.call(inputs=(u_ids, i_ids), training=False, is_rec=True, **kwargs)

        # e_var = self.paddingItems.lookup(tf.squeeze(tf.cast(i_ids, tf.int32)))
        # u_e = self.user_embeddings(u_ids)
        # i_e = self.item_embeddings(i_ids)
        # e_e = self.ent_embeddings(e_var)
        # ie_e = i_e + e_e
        #
        # _, r_e, norm = self.getPreferences(u_e, ie_e)
        #
        # proj_u_e = self.projection_trans_h(u_e, norm)
        # proj_i_e = self.projection_trans_h(ie_e, norm)
        #
        # if self.L1_flag:
        #     score = tf.reduce_sum(tf.abs(proj_u_e + r_e - proj_i_e), -1)
        # else:
        #     score = tf.reduce_sum((proj_u_e + r_e - proj_i_e) ** 2, -1)
        return tf.squeeze(score)

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)