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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(0)


class KGFlexTFModel(keras.Model):
    def __init__(self,
                 num_users,
                 num_items,
                 user_feature_weights,
                 user_item_features,
                 num_features,
                 factors=10,
                 learning_rate=0.01,
                 name="KGFlex_TF",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(42)
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self._factors = factors

        self.initializer = tf.initializers.RandomNormal(stddev=0.1)

        self.K = user_feature_weights
        self.B = tf.Variable(self.initializer(shape=[self.num_features]), name='B', dtype=tf.float32)
        self.H = tf.Variable(self.initializer(shape=[self.num_users, self._factors]), name='H', dtype=tf.float32)
        self.G = tf.Variable(self.initializer(shape=[self.num_features, self._factors]), name='G', dtype=tf.float32)
        self.C = user_item_features

        self.optimizer = tf.optimizers.Adam(learning_rate)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        user, item = inputs
        h_u = tf.squeeze(tf.nn.embedding_lookup(self.H, user))
        z_u = h_u @ tf.transpose(self.G)  # num_features x 1
        k_u = tf.squeeze(tf.nn.embedding_lookup(self.K, user))  # num_features x 1
        a_u = k_u * (tf.add(z_u, self.B))
        ui_pairs = tf.stack([tf.squeeze(user), tf.squeeze(item)], axis=-1)
        features = tf.gather_nd(self.C, ui_pairs)
        x_ui = tf.reduce_sum(tf.gather(a_u, features, batch_dims=1), axis=-1)

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
            # reg_loss = self._l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
            #                                      tf.nn.l2_loss(gamma_pos),
            #                                      tf.nn.l2_loss(gamma_neg)]) \
            #            + self._l_b * tf.nn.l2_loss(beta_pos) \
            #            + self._l_b * tf.nn.l2_loss(beta_neg) / 10

            # Loss to be optimized
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
        Z = self.H @ tf.transpose(self.G)
        A = self.K * Z
        predictions = tf.reduce_sum(
            tf.gather(tf.gather(A, list(range(self.num_users))), tf.gather(self.C, list(range(self.num_users))),
                      batch_dims=1), axis=-1).to_tensor()
        return predictions

    def get_all_topks(self, predictions, mask, k, user_map, item_map):
        predictions_top_k = {
            user_map[u]: list(map(lambda x: (item_map.get(x[0]), x[1]), zip(*map(lambda x: x.numpy(), top[::-1])))) for
            u, top in enumerate(zip(*tf.nn.top_k(tf.where(mask, predictions, -np.inf), k=k)))}
        return predictions_top_k
