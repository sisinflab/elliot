"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, daniele.malitesta@poliba.it'

import tensorflow as tf
import numpy as np
from tensorflow import keras


class AMR_model(keras.Model):
    def __init__(self, factors=200, factors_d=20,
                 learning_rate=0.001,
                 l_w=0, l_b=0, l_e=0, eps=0, l_adv=0,
                 emb_image=None,
                 num_users=100,
                 num_items=100,
                 name="AMR",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(42)

        self._factors = factors
        self._factors_d = factors_d
        self._learning_rate = learning_rate
        self.l_w = l_w
        self.l_b = l_b
        self.l_e = l_e
        self._eps = eps
        self._l_adv = l_adv
        self.emb_image = emb_image
        self.num_image_feature = self.emb_image.shape[1]
        self._num_items = num_items
        self._num_users = num_users

        self.initializer = tf.initializers.GlorotUniform()

        self.Bi = tf.Variable(tf.zeros(self._num_items), name='Bi', dtype=tf.float32)
        self.Gu = tf.Variable(self.initializer(shape=[self._num_users, self._factors]), name='Gu', dtype=tf.float32)
        self.Gi = tf.Variable(self.initializer(shape=[self._num_items, self._factors]), name='Gi', dtype=tf.float32)

        self.Bp = tf.Variable(
            self.initializer(shape=[self.num_image_feature, 1]), name='Bp', dtype=tf.float32)
        self.Tu = tf.Variable(
            self.initializer(shape=[self._num_users, self._factors_d]),
            name='Tu', dtype=tf.float32)
        self.F = tf.Variable(
            self.emb_image, dtype=tf.float32, trainable=True)
        self.E = tf.Variable(
            self.initializer(shape=[self.num_image_feature, self._factors_d]),
            name='E', dtype=tf.float32)

        # Initialize the perturbation with 0 values
        self._Delta_F = tf.Variable(tf.zeros(shape=[self._num_items, self.num_image_feature]), dtype=tf.float32, trainable=False)

        self.optimizer = tf.optimizers.Adam(self._learning_rate)
        # self.saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

    #@tf.function
    def call(self, inputs, training=None, mask=None):
        user, item = inputs
        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        theta_u = tf.squeeze(tf.nn.embedding_lookup(self.Tu, user))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        feature_i = tf.squeeze(tf.nn.embedding_lookup(self.F, item)) + tf.squeeze(tf.nn.embedding_lookup(self._Delta_F, item))

        xui = beta_i + tf.reduce_sum((gamma_u * gamma_i), axis=1) + \
              tf.reduce_sum((theta_u * tf.matmul(feature_i, self.E)), axis=1) + \
              tf.squeeze(tf.matmul(feature_i, self.Bp))

        return xui, gamma_u, gamma_i, feature_i, theta_u, beta_i

    #@tf.function
    def train_step(self, batch, user_adv_train=False):
        user, pos, neg = batch
        with tf.GradientTape() as t:
            xu_pos, gamma_u, gamma_pos, _, theta_u, beta_pos = \
                self(inputs=(user, pos), training=True)
            xu_neg, _, gamma_neg, _, _, beta_neg = self(inputs=(user, neg), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg),
                                                 tf.nn.l2_loss(theta_u)]) \
                       + self.l_b * tf.nn.l2_loss(beta_pos) \
                       + self.l_b * tf.nn.l2_loss(beta_neg) / 10 \
                       + self.l_e * tf.reduce_sum([tf.nn.l2_loss(self.E), tf.nn.l2_loss(self.Bp)])

            # Loss to be optimized
            loss += reg_loss

            if user_adv_train:
                # Build the Adversarial Perturbation on the Current Model Parameters
                self.build_perturbation(batch)

                # Clean Inference
                adv_xu_pos, _, _, _, _, _ = self(inputs=(user, pos), training=True)
                adv_xu_neg, _, _, _, _, _ = self(inputs=(user, neg), training=True)

                adv_difference = tf.clip_by_value(adv_xu_pos - adv_xu_neg, -80.0, 1e8)
                adv_loss = tf.reduce_sum(tf.nn.softplus(-adv_difference))

                loss += self._l_adv * adv_loss

        grads = t.gradient(loss, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp])
        self.optimizer.apply_gradients(zip(grads, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp]))

        return loss

    #@tf.function
    def predict(self, start, stop):
        return self.Bi + tf.matmul(self.Gu[start:stop], self.Gi, transpose_b=True) \
               + tf.matmul(self.Tu[start:stop], tf.matmul(self.F, self.E), transpose_b=True) \
               + tf.squeeze(tf.matmul(self.F, self.Bp))

    def get_config(self):
        raise NotImplementedError

    #@tf.function
    def build_perturbation(self, batch):
        """
        Evaluate Adversarial Perturbation with FGSM-like Approach
        """
        self._Delta_F = self._Delta_F * 0.0

        user, pos, neg = batch
        with tf.GradientTape() as tape_adv:
            xu_pos, gamma_u, gamma_pos, _, theta_u, beta_pos = \
                self(inputs=(user, pos), training=True)
            xu_neg, _, gamma_neg, _, _, beta_neg = self(inputs=(user, neg), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg),
                                                 tf.nn.l2_loss(theta_u)]) \
                       + self.l_b * tf.nn.l2_loss(beta_pos) \
                       + self.l_b * tf.nn.l2_loss(beta_neg) / 10 \
                       + self.l_e * tf.reduce_sum([tf.nn.l2_loss(self.E), tf.nn.l2_loss(self.Bp)])

            # Loss to be optimized
            loss += reg_loss

        grad_F = tape_adv.gradient(loss, [self.F])
        grad_F = tf.stop_gradient(grad_F[0])

        self._Delta_F = tf.nn.l2_normalize(grad_F, 1) * self._eps

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
