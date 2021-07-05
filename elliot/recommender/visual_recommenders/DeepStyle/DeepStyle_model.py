"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import tensorflow as tf
from tensorflow import keras

import numpy as np


class DeepStyleModel(keras.Model):
    def __init__(self,
                 factors=20,
                 learning_rate=0.001,
                 l_w=0,
                 num_image_feature=2048,
                 num_users=100,
                 num_items=100,
                 name="DeepStyle",
                 random_seed=42,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        tf.random.set_seed(random_seed)

        self._factors = factors
        self._learning_rate = learning_rate
        self.l_w = l_w
        self._num_image_feature = num_image_feature
        self._num_items = num_items
        self._num_users = num_users

        self.initializer = tf.initializers.GlorotUniform()

        self.Gu = tf.Variable(self.initializer(shape=[self._num_users, self._factors]), name='Gu', dtype=tf.float32)
        self.Gi = tf.Variable(self.initializer(shape=[self._num_items, self._factors]), name='Gi', dtype=tf.float32)
        self.Li = tf.Variable(
            self.initializer(shape=[self._num_items, self._factors]),
            name='Li', dtype=tf.float32)
        self.E = tf.Variable(
            self.initializer(shape=[self._num_image_feature, self._factors]),
            name='E', dtype=tf.float32)

        self.optimizer = tf.optimizers.Adam(self._learning_rate)

    @tf.function
    def call(self, inputs, training=None):
        user, item, feature_i = inputs
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))

        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))

        l_i = tf.squeeze(tf.nn.embedding_lookup(self.Li, item))

        xui = tf.reduce_sum(gamma_u * (tf.matmul(feature_i, self.E) - l_i + gamma_i), 1)

        return xui, gamma_u, gamma_i, feature_i, l_i

    @tf.function
    def train_step(self, batch):
        user, pos, feat_pos, neg, feat_neg = batch
        with tf.GradientTape() as t:
            # Clean Inference
            xu_pos, gamma_u, gamma_pos, _, l_pos = self.call(inputs=(user, pos, feat_pos), training=True)
            xu_neg, _, gamma_neg, _, l_neg = self.call(inputs=(user, neg, feat_neg), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg),
                                                 tf.nn.l2_loss(l_pos),
                                                 tf.nn.l2_loss(l_neg)])

            # Loss to be optimized
            loss += reg_loss

        grads = t.gradient(loss, [self.Gu, self.Gi, self.E, self.Li])
        self.optimizer.apply_gradients(zip(grads, [self.Gu, self.Gi, self.E, self.Li]))

        return loss

    @tf.function
    def predict_batch(self, start, stop, gi, li, fi):
        return tf.reduce_sum(self.Gu[start:stop] * (tf.matmul(fi, self.E) - li + gi), axis=1)

    @tf.function
    def predict_item_batch(self, start, stop, start_item, stop_item, feat):
        return tf.matmul(self.Gu[start:stop], (tf.matmul(feat, self.E) - self.Li[start_item:(stop_item + 1)]
                                               + self.Gi[start_item:(stop_item + 1)]), transpose_b=True)

    @tf.function
    def get_config(self):
        raise NotImplementedError

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
