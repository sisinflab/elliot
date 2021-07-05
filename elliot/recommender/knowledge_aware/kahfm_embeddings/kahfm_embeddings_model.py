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


class KaHFMEmbeddingsModel(keras.Model):

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

        self.initializer = tf.initializers.GlorotUniform()

        self.user_embedding = keras.layers.Embedding(input_dim=user_factors.shape[0], output_dim=user_factors.shape[1],
                                                     weights=[user_factors],
                                                     embeddings_regularizer=keras.regularizers.l2(self.l_w),
                                                     trainable=True, dtype=tf.float32)
        self.item_embedding = keras.layers.Embedding(input_dim=item_factors.shape[0], output_dim=item_factors.shape[1],
                                                     weights=[item_factors],
                                                     embeddings_regularizer=keras.regularizers.l2(self.l_w),
                                                     trainable=True, dtype=tf.float32)
        self.item_bias_embedding = keras.layers.Embedding(input_dim=item_factors.shape[0], output_dim=1,
                                                          embeddings_initializer=self.initializer,
                                                          embeddings_regularizer=keras.regularizers.l2(
                                                              self.l_b),
                                                          dtype=tf.float32)
        self.user_embedding(0)
        self.item_embedding(0)
        self.item_bias_embedding(0)

        self.optimizer = tf.optimizers.Adam(self._learning_rate)
        #self.saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        user, item = inputs
        beta_i = self.item_bias_embedding(tf.squeeze(item))
        gamma_u = self.user_embedding(tf.squeeze(user))
        gamma_i = self.item_embedding(tf.squeeze(item))

        xui = beta_i + tf.reduce_sum(gamma_u * gamma_i, -1)

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
            # # Regularization Component
            # reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
            #                                      tf.nn.l2_loss(gamma_pos),
            #                                      tf.nn.l2_loss(gamma_neg)]) \
            #            + self.l_b * tf.nn.l2_loss(beta_pos) \
            #            + self.l_b * tf.nn.l2_loss(beta_neg) / 10
            #
            # # Loss to be optimized
            # loss += reg_loss

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss

    @tf.function
    def predict_batch(self, start, stop):
        return tf.transpose(self.item_bias_embedding.weights[0]) + tf.matmul(self.user_embedding.weights[0][start:stop], self.item_embedding.weights[0], transpose_b=True)

    @tf.function
    def predict(self, inputs, training=False, **kwargs):
        logits, _ = self.call(inputs=inputs, training=True)
        return logits

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)

    def get_config(self):
        raise NotImplementedError
