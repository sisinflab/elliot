"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it,daniele.malitesta@poliba.it'

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class LogisticMatrixFactorizationModel(keras.Model):
    def __init__(self,
                 num_users,
                 num_items,
                 factors,
                 lambda_weights,
                 alpha,
                 learning_rate=0.01,
                 random_seed=42,
                 name="LMF",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)
        self._num_users = num_users
        self._num_items = num_items
        self._factors = factors
        self._lambda_weights = lambda_weights
        self._alpha = alpha
        self._user_update = False

        self.initializer = tf.initializers.GlorotUniform()

        self.Gu = tf.Variable(self.initializer(shape=[self._num_users, self._factors]), name='Gu', dtype=tf.float32)
        self.Gi = tf.Variable(self.initializer(shape=[self._num_items, self._factors]), name='Gi', dtype=tf.float32)
        self.Bu = tf.Variable(tf.zeros(self._num_users), name='Bu', dtype=tf.float32)
        self.Bi = tf.Variable(tf.zeros(self._num_items), name='Bi', dtype=tf.float32)

        self.optimizer = tf.optimizers.Adagrad(learning_rate)

    @tf.function
    def set_update_user(self, update_user):
        self._user_update = update_user

    @tf.function
    def call(self, inputs, training=None, mask=None):
        user, item = inputs

        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        beta_u = tf.squeeze(tf.nn.embedding_lookup(self.Bu, user))
        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))

        xui = tf.reduce_sum(gamma_u * gamma_i, -1) + beta_u + beta_i

        return xui, gamma_u, gamma_i, beta_u, beta_i

    @tf.function
    def train_step(self, batch):
        user, pos, label = batch

        with tf.GradientTape() as tape:
            # Clean Inference
            output, g_u, g_i, b_u, b_i = self(inputs=(user, pos), training=True)

            label = tf.dtypes.cast(label, tf.float32)

            # We want to maximize the log posterior
            loss = tf.reduce_sum(-(self._alpha * label * output - (1 + self._alpha * label) * tf.math.log(1 + tf.math.exp(output))))

            # Regularization Component
            reg_loss = self._lambda_weights * tf.reduce_sum([tf.nn.l2_loss(g_u),
                                                             tf.nn.l2_loss(g_i)])

            loss += reg_loss

        if self._user_update:
            grads = tape.gradient(loss, [self.Gu, self.Bu])
            self.optimizer.apply_gradients(zip(grads, [self.Gu, self.Bu]))
        else:
            grads = tape.gradient(loss, [self.Gi, self.Bi])
            self.optimizer.apply_gradients(zip(grads, [self.Gi, self.Bi]))

        return loss

    @tf.function
    def predict_batch(self, start, stop, **kwargs):
        return tf.expand_dims(self.Bu[start:stop], -1) + tf.transpose(tf.expand_dims(self.Bi, -1)) + tf.matmul(self.Gu[start:stop], self.Gi, transpose_b=True)

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
