"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import os

import tensorflow as tf
from tensorflow import keras

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.random.set_seed(42)


class DeepStyle_model(keras.Model):
    def __init__(self,
                 factors=20,
                 learning_rate=0.001,
                 l_w=0,
                 emb_image=None,
                 num_users=100,
                 num_items=100,
                 name="DeepStyle",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self._factors = factors
        self._learning_rate = learning_rate
        self.l_w = l_w
        self._emb_image = emb_image
        self._num_image_feature = self._emb_image.shape[1]
        self._num_items = num_items
        self._num_users = num_users

        self.initializer = tf.initializers.GlorotUniform()

        self.Gu = tf.Variable(self.initializer(shape=[self._num_users, self._factors]), name='Gu', dtype=tf.float32)
        self.Gi = tf.Variable(self.initializer(shape=[self._num_items, self._factors]), name='Gi', dtype=tf.float32)
        self.L = tf.Variable(
            self.initializer(shape=[self._num_items, self._factors]),
            name='L', dtype=tf.float32)
        self.F = tf.Variable(
            self._emb_image,
            name='F', dtype=tf.float32, trainable=False)
        self.E = tf.Variable(
            self.initializer(shape=[self._num_image_feature, self._factors]),
            name='E', dtype=tf.float32)

        self.optimizer = tf.optimizers.Adam(self._learning_rate)

    @tf.function
    def call(self, inputs, training=None):
        user, item = inputs
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))

        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        feature_i = tf.squeeze(tf.nn.embedding_lookup(self.F, item))

        l_i = tf.squeeze(tf.nn.embedding_lookup(self.L, item))

        xui = tf.reduce_sum(gamma_u * (tf.matmul(feature_i, self.E) - l_i + gamma_i), 1)

        return xui, gamma_u, gamma_i, feature_i, l_i

    @tf.function
    def train_step(self, batch):
        user, pos, neg = batch
        with tf.GradientTape() as t:
            # Clean Inference
            xu_pos, gamma_u, gamma_pos, _, l_pos = self.call(inputs=(user, pos), training=True)
            xu_neg, _, gamma_neg, _, l_neg = self.call(inputs=(user, neg), training=True)

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

        grads = t.gradient(loss, [self.Gu, self.Gi, self.E, self.L])
        self.optimizer.apply_gradients(zip(grads, [self.Gu, self.Gi, self.E, self.L]))

        return loss

    @tf.function
    def predict(self, start, stop, training=False):
        return tf.matmul(self.Gu[start:stop], (tf.matmul(self.F, self.E) - self.L + self.Gi), transpose_b=True)

    @tf.function
    def get_config(self):
        raise NotImplementedError

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
