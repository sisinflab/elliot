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


class KaHFM_model(keras.Model):

    def __init__(self,
                 user_factors,
                 item_factors,
                 learning_rate=0.001,
                 l_w=0, l_b=0,
                 random_seed=42,
                 name="NNBPRMF",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        # self._factors = factors
        self._learning_rate = learning_rate
        self.l_w = l_w
        self.l_b = l_b
        # self._num_items = num_items
        # self._num_users = num_users

        self.initializer = tf.initializers.GlorotUniform()
        self.Bi = tf.Variable(tf.zeros(item_factors.shape[0]), name='Bi', dtype=tf.float32)
        self.Gu = tf.Variable(user_factors, name='Gu', dtype=tf.float32)
        self.Gi = tf.Variable(item_factors, name='Gi', dtype=tf.float32)

        self.optimizer = tf.optimizers.Adam(self._learning_rate)
        # self.saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        user, item = inputs
        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))

        xui = beta_i + tf.reduce_sum(gamma_u * gamma_i, 1)

        return xui, beta_i, gamma_u, gamma_i

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            user, pos, neg = batch
            # Clean Inference
            xu_pos, beta_pos, gamma_u, gamma_pos = self.call(inputs=(user, pos), training=True)
            xu_neg, beta_neg, gamma_u, gamma_neg = self.call(inputs=(user, neg), training=True)

            difference = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-difference))
            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg)]) \
                       + self.l_b * tf.nn.l2_loss(beta_pos) \
                       + self.l_b * tf.nn.l2_loss(beta_neg) / 10

            # Loss to be optimized
            loss += reg_loss

        grads = tape.gradient(loss, [self.Bi, self.Gu, self.Gi])
        self.optimizer.apply_gradients(zip(grads, [self.Bi, self.Gu, self.Gi]))

        return loss

    @tf.function
    def predict_all(self):
        return self.Bi + tf.matmul(self.Gu, self.Gi, transpose_b=True)

    @tf.function
    def predict_batch(self, start, stop):
        return self.Bi + tf.matmul(self.Gu[start:stop], self.Gi, transpose_b=True)

    @tf.function
    def predict(self, inputs, training=False, **kwargs):
        logits, _ = self.call(inputs=inputs, training=True)
        return logits

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)

    def get_config(self):
        raise NotImplementedError
