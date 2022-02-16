"""
Module description:
"""

__version__ = '0.1'
__author__ = 'Antonio Ferrara'
__email__ = 'antonio.ferrara@poliba.it'

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.sparse.linalg import LinearOperator
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(0)


class KGFlexModel(keras.Model):
    def __init__(self,
                 num_users,
                 num_items,
                 user_feature_weights,
                 content_vectors,
                 num_features,
                 factors=10,
                 learning_rate=0.01,
                 l_w=0.1, l_b=0.001,
                 name="KGFlex_TF",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(42)
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self._factors = factors
        self._l_w = l_w
        self._l_b = l_b

        self.initializer = tf.initializers.RandomNormal(stddev=0.1)

        self.K = user_feature_weights
        self.F_B = tf.Variable(self.initializer(shape=[self.num_features]), name='F_B', dtype=tf.float32)
        self.I_B = tf.Variable(self.initializer(shape=[self.num_items]), name='I_B', dtype=tf.float32)
        # self.U_B = tf.Variable(self.initializer(shape=[self.num_users]), name='U_B', dtype=tf.float32)
        self.H = tf.Variable(self.initializer(shape=[self.num_users, self._factors]), name='H', dtype=tf.float32)
        self.G = tf.Variable(self.initializer(shape=[self.num_features, self._factors]), name='G', dtype=tf.float32)
        self.C = content_vectors

        self.optimizer = tf.optimizers.Adam(learning_rate)

    def scipy_gather(self, idx):
        return self.C[idx].A - 1

    def scipy_matmul(self, mat):
        def subtract_and_matvec(x):
            return self.C * x - 1 * x.sum()

        op = LinearOperator(self.C.shape, matvec=subtract_and_matvec)

        return op.dot(mat)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        user, item = inputs
        i_b = tf.squeeze(tf.nn.embedding_lookup(self.I_B, item))
        h_u = tf.squeeze(tf.nn.embedding_lookup(self.H, user))
        z_u = tf.add(h_u @ tf.transpose(self.G), self.F_B)  # num_features x 1
        c_i = tf.py_function(self.scipy_gather, [tf.squeeze(item)], Tout=[tf.float32])
        s_ui = c_i * z_u
        k_u = tf.squeeze(tf.nn.embedding_lookup(self.K, user))  # num_features x 1
        inter_ui = k_u * s_ui
        x_ui = tf.add(tf.reduce_sum(inter_ui, axis=-1), i_b)
        # x_ui = tf.reduce_sum(inter_ui, axis=-1)
        # x_ui = tf.reduce_sum(inter_ui, axis=-1)

        return x_ui

    @tf.function
    def train_step(self, batch):
        user, pos, neg = batch
        with tf.GradientTape() as tape:
            # Clean Inference
            xu_pos = self(inputs=(user, pos), training=True)
            xu_neg = self(inputs=(user, neg), training=True)

            difference = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-difference))
            # Regularization Component
            # reg_loss = self._l_w * tf.reduce_sum([tf.nn.l2_loss(inter_pos),
            #                                       tf.nn.l2_loss(inter_neg)])
                #        + self._l_b * tf.nn.l2_loss(
                # i_b_pos) + self._l_b * tf.nn.l2_loss(i_b_neg) / 10

            # # Loss to be optimized
            # loss += reg_loss

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss

    @tf.function
    def predict(self, inputs, training=False, **kwargs):
        """
        Get full predictions on the whole users/items matrix.
        Returns:
            The matrix of predicted values.
        """
        output = self.call(inputs=inputs, training=training)
        return output

    @tf.function
    def get_all_recs(self):
        # Z = self.H @ tf.transpose(self.G)
        Z = tf.add((self.H @ tf.transpose(self.G)), self.F_B)
        # return tf.add(tf.transpose(
        #     tf.squeeze(tf.py_function(self.scipy_matmul, [tf.transpose(self.K * Z)], Tout=[tf.float32]))), self.I_B)
        # result = tf.transpose(
        #     tf.squeeze(tf.py_function(self.scipy_matmul, [tf.transpose(self.K * Z)], Tout=[tf.float32])))
        result = tf.add(tf.transpose(
            tf.squeeze(tf.py_function(self.scipy_matmul, [tf.transpose(self. K * Z)], Tout=[tf.float32]))), self.I_B)
        return result

    def get_all_topks(self, predictions, mask, k, user_map, item_map):
        predictions_top_k = {
            user_map[u]: list(map(lambda x: (item_map.get(x[0]), x[1]), zip(*map(lambda x: x.numpy(), top[::-1])))) for
            u, top in enumerate(zip(*tf.nn.top_k(tf.where(mask, predictions, -np.inf), k=k)))}
        return predictions_top_k
