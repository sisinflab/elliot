"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import numpy as np
import tensorflow as tf
from tensorflow import keras

from elliot.recommender.visual_recommenders.DVBPR.FeatureExtractor import FeatureExtractor


class DVBPRModel(keras.Model):
    def __init__(self,
                 factors=200,
                 learning_rate=0.001,
                 lambda_1=0,
                 lambda_2=0,
                 num_users=100,
                 num_items=100,
                 random_seed=42,
                 name="DVBPR",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self._factors = factors
        self._learning_rate = learning_rate
        self._lambda_1 = lambda_1
        self._lambda_2 = lambda_2
        self._num_items = num_items
        self._num_users = num_users

        self.initializer = tf.initializers.GlorotUniform()

        self.Tu = tf.Variable(self.initializer(shape=[self._num_users, self._factors]), name='Tu', dtype=tf.float32)
        self.Cnn = FeatureExtractor(self._factors)

        self.optimizer = tf.optimizers.Adam(self._learning_rate)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        user, im = inputs
        cnn_output = self.Cnn(inputs=im, training=training)
        theta_u = tf.squeeze(tf.nn.embedding_lookup(self.Tu, user))

        xui = tf.reduce_sum(theta_u * cnn_output, 1)

        return xui, theta_u, cnn_output

    @tf.function
    def train_step(self, batch):
        user, pos, pos_im, neg, neg_im = batch

        with tf.GradientTape() as tape:
            # Clean Inference
            xu_pos, theta_u, _ = self(inputs=(user, pos_im), training=True)
            xu_neg, _, _ = self(inputs=(user, neg_im), training=True)

            difference = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-difference))

            # Regularization Component
            reg_loss = self._lambda_1 * tf.nn.l2_loss(theta_u) \
                       + self._lambda_2 * tf.reduce_sum([tf.nn.l2_loss(layer)
                                                         for layer in self.Cnn.trainable_variables
                                                         if 'bias' not in layer.name])

            # Loss to be optimized
            loss += reg_loss

        params = [self.Tu,
                  *self.Cnn.trainable_variables]

        grads = tape.gradient(loss, params)
        self.optimizer.apply_gradients(zip(grads, params))

        return loss

    @tf.function
    def predict_item_batch(self, start, stop, phi):
        return tf.matmul(self.Tu[start:stop], phi, transpose_b=True)

    @tf.function
    def get_config(self):
        raise NotImplementedError

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
