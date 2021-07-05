"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class CML_model(keras.Model):

    def __init__(self,
                 user_factors=200,
                 item_factors=200,
                 learning_rate=0.001,
                 l_w=0, l_b=0, margin=0.5,
                 num_users=100,
                 num_items=100,
                 random_seed=42,
                 name="CML",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self._user_factors = user_factors
        self._item_factors = item_factors
        self._learning_rate = learning_rate
        self.l_w = l_w
        self.l_b = l_b
        self._num_items = num_items
        self._num_users = num_users

        self.initializer = tf.initializers.GlorotUniform()

        self.Gu = LatentFactor(num_instances=self._num_users,
                                                dim=self._user_factors,
                                                name='user_latent_factor')
        self.Gi = LatentFactor(num_instances=self._num_items,
                                                dim=self._item_factors,
                                                name='item_latent_factor')

        self.Bi = LatentFactor(num_instances=self._num_items,
                                       dim=1,
                                       name='item_bias')

        self.margin = margin

        self.optimizer = tf.optimizers.Adam(self._learning_rate)

    @tf.function
    def call(self, inputs, training=None):
        user, item = inputs
        beta_i = tf.squeeze(self.Bi(item))
        gamma_u = tf.squeeze(self.Gu(user))
        gamma_i = tf.squeeze(self.Gi(item))

        l2_user_pos = tf.math.reduce_sum(tf.math.square(gamma_u - gamma_i),
                                         axis=-1,
                                         keepdims=True)

        score = (-l2_user_pos) + beta_i

        return score, beta_i, gamma_u, gamma_i

    @tf.function
    def train_step(self, batch):
        user, pos, neg = batch
        with tf.GradientTape() as tape:
            # Clean Inference
            xu_pos, beta_pos, gamma_u, gamma_pos = self(inputs=(user, pos), training=True)
            xu_neg, beta_neg, gamma_u, gamma_neg = self(inputs=(user, neg), training=True)

            difference = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.maximum(self.margin - difference, 0))
            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg)]) \
                       + self.l_b * tf.nn.l2_loss(beta_pos) \
                       + self.l_b * tf.nn.l2_loss(beta_neg) / 10

            # Loss to be optimized
            loss += reg_loss

        # grads = tape.gradient(loss, [self.Bi, self.Gu, self.Gi])
        # self.optimizer.apply_gradients(zip(grads, [self.Bi, self.Gu, self.Gi]))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss

    @tf.function
    def predict(self, start, stop,  **kwargs):
        # return self.Bi + tf.matmul(self.Gu[start:stop], self.Gi, transpose_b=True)
        user_vec = self.Gu.embeddings[start:stop]
        return -tf.math.reduce_sum(
            tf.math.square(tf.expand_dims(user_vec, axis=1) - self.Gi.variables[0]), axis=-1,
            keepdims=False) + tf.reshape(self.Bi.variables[0], [-1])

    @tf.function
    def get_top_k(self, predictions, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, predictions, -np.inf), k=k, sorted=True)

    @tf.function
    def get_positions(self, predictions, train_mask, items, inner_test_user_true_mask):
        predictions = tf.gather(predictions, inner_test_user_true_mask)
        train_mask = tf.gather(train_mask, inner_test_user_true_mask)
        equal = tf.reshape(items, [len(items), 1])
        i = tf.argsort(tf.where(train_mask, predictions, -np.inf), axis=-1,
                       direction='DESCENDING', stable=False, name=None)
        positions = tf.where(tf.equal(equal, i))[:, 1]
        return 1 - (positions / tf.reduce_sum(tf.cast(train_mask, tf.int64), axis=1))

    def get_config(self):
        raise NotImplementedError


class LatentFactor(tf.keras.layers.Embedding):

    def __init__(self, num_instances, dim, zero_init=False, name=None):

        if zero_init:
            initializer = 'zeros'
        else:
            initializer = 'uniform'
        super(LatentFactor, self).__init__(input_dim=num_instances,
                                           output_dim=dim,
                                           embeddings_initializer=initializer,
                                           name=name)

    def censor(self, censor_id):

        unique_censor_id, _ = tf.unique(censor_id)
        embedding_gather = tf.gather(self.variables[0], indices=unique_censor_id)
        norm = tf.norm(embedding_gather, axis=1, keepdims=True)
        return self.variables[0].scatter_nd_update(indices=tf.expand_dims(unique_censor_id, 1),
                                                   updates=embedding_gather / tf.math.maximum(norm, 0.1))
