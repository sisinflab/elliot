"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import numpy as np
import tensorflow as tf
from tensorflow import keras, Variable


class AMR_model(keras.Model):

    def __init__(self, factors=200, factors_d=20,
                 learning_rate=0.001,
                 l_w=0, l_b=0, l_e=0,
                 num_image_feature=2048,
                 num_users=100,
                 num_items=100,
                 eps=0.05,
                 l_adv=0.001,
                 batch_size=256,
                 random_seed=42,
                 name="AMR",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self.factors = factors
        self.factors_d = factors_d
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.l_b = l_b
        self.l_e = l_e
        self.num_image_feature = num_image_feature
        self.num_items = num_items
        self.num_users = num_users
        self.l_adv = l_adv
        self.eps = eps
        self.batch_size = batch_size
        self.initializer = tf.initializers.GlorotUniform()

        self.Bi = tf.Variable(tf.zeros(self.num_items), name='Bi', dtype=tf.float32)
        self.Gu = tf.Variable(self.initializer(shape=[self.num_users, self.factors]), name='Gu', dtype=tf.float32)
        self.Gi = tf.Variable(self.initializer(shape=[self.num_items, self.factors]), name='Gi', dtype=tf.float32)

        self.Bp = tf.Variable(
            self.initializer(shape=[self.num_image_feature, 1]), name='Bp', dtype=tf.float32)
        self.Tu = tf.Variable(
            self.initializer(shape=[self.num_users, self.factors_d]),
            name='Tu', dtype=tf.float32)
        self.E = tf.Variable(
            self.initializer(shape=[self.num_image_feature, self.factors_d]),
            name='E', dtype=tf.float32)

        # Temporal to have better performance
        self.Delta_F = tf.Variable(tf.zeros(shape=[self.batch_size, self.num_image_feature]), dtype=tf.float32,
                                   trainable=True)

        self.optimizer = tf.optimizers.Adam(self.learning_rate)

    def call(self, inputs, adversarial=False, training=None):
        user, item, feature_i = inputs
        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        theta_u = tf.squeeze(tf.nn.embedding_lookup(self.Tu, user))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))

        if adversarial:
            feature_i = feature_i + self.Delta_F

        xui = beta_i + tf.reduce_sum((gamma_u * gamma_i), axis=1) + \
              tf.reduce_sum((theta_u * tf.matmul(feature_i, self.E)), axis=1) + \
              tf.squeeze(tf.matmul(feature_i, self.Bp))

        return xui, gamma_u, gamma_i, feature_i, theta_u, beta_i

    def train_step(self, batch, use_adv_train=False):
        user, pos, feature_pos, neg, feature_neg = batch
        with tf.GradientTape() as t:
            xu_pos, gamma_u, gamma_pos, _, theta_u, beta_pos = \
                self(inputs=(user, pos, feature_pos), training=True)
            xu_neg, _, gamma_neg, _, _, beta_neg = self(inputs=(user, neg, feature_neg), adversarial=False,
                                                        training=True)

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

            if use_adv_train:
                # Build the Adversarial Perturbation on the Current Model Parameters
                self.build_perturbation(batch)

                # Clean Inference
                adv_xu_pos, _, _, _, _, _ = self(inputs=(user, pos, feature_pos), adversarial=True, training=True)
                adv_xu_neg, _, _, _, _, _ = self(inputs=(user, neg, feature_neg), adversarial=False, training=True)

                adv_difference = tf.clip_by_value(adv_xu_pos - adv_xu_neg, -80.0, 1e8)
                adv_loss = tf.reduce_sum(tf.nn.softplus(-adv_difference))

                loss += self.l_adv * adv_loss

        grads = t.gradient(loss, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp])
        self.optimizer.apply_gradients(zip(grads, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp]))

        return loss

    def predict_item_batch(self, start, stop, start_item, stop_item, feat, delta_features):
        if delta_features is None:
            return self.Bi[start_item:(stop_item + 1)] + tf.matmul(self.Gu[start:stop],
                                                                   self.Gi[start_item:(stop_item + 1)],
                                                                   transpose_b=True) \
                   + tf.matmul(self.Tu[start:stop], tf.matmul(feat, self.E), transpose_b=True) \
                   + tf.squeeze(tf.matmul(feat, self.Bp))
        else:
            return self.Bi[start_item:(stop_item + 1)] + tf.matmul(self.Gu[start:stop],
                                                                   self.Gi[start_item:(stop_item + 1)],
                                                                   transpose_b=True) \
                   + tf.matmul(self.Tu[start:stop],
                               tf.matmul(feat + delta_features[start_item:(stop_item + 1)], self.E), transpose_b=True) \
                   + tf.squeeze(tf.matmul(feat + delta_features[start_item:(stop_item + 1)], self.Bp))

    def get_config(self):
        raise NotImplementedError

    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)

    def init_delta_f(self):
        self.Delta_F = tf.Variable(tf.zeros(shape=[self.batch_size, self.num_image_feature]), dtype=tf.float32,
                                   trainable=True)

    def build_perturbation(self, batch, delta_f=None):
        """
        Evaluate Adversarial Perturbation with FGSM-like Approach
        """
        user, pos, feature_pos, neg, feature_neg = batch

        if delta_f is not None:
            self.Delta_F = tf.Variable(delta_f, trainable=True)

        with tf.GradientTape() as tape_adv:
            # Clean Inference
            xu_pos, gamma_u, gamma_pos, _, theta_u, beta_pos = \
                self(inputs=(user, pos, feature_pos), adversarial=True, training=True)
            xu_neg, _, gamma_neg, _, _, beta_neg = self(inputs=(user, neg, feature_neg), adversarial=False,
                                                        training=True)

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

        grad_F = tape_adv.gradient(loss, self.Delta_F)
        grad_F = tf.stop_gradient(grad_F)
        self.Delta_F = tf.Variable(tf.nn.l2_normalize(grad_F, 1) * self.eps)
        return self.Delta_F

    def build_msap_perturbation(self, batch, eps_iter, nb_iter, delta_f=None):
        """
        Evaluate Adversarial Perturbation with MSAP
        https://journals.flvc.org/FLAIRS/article/view/128443
        """
        user, pos, feature_pos, neg, feature_neg = batch

        if delta_f is not None:
            self.Delta_F = tf.Variable(delta_f, trainable=True)

        for _ in range(nb_iter):
            with tf.GradientTape() as tape_adv:
                # Clean Inference
                xu_pos, gamma_u, gamma_pos, _, theta_u, beta_pos = \
                    self(inputs=(user, pos, feature_pos), adversarial=True, training=True)
                xu_neg, _, gamma_neg, _, _, beta_neg = self(inputs=(user, neg, feature_neg), adversarial=False,
                                                            training=True)

                difference = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
                loss = tf.reduce_sum(tf.nn.softplus(-difference))
                # Regularization Component
                reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                     tf.nn.l2_loss(gamma_pos),
                                                     tf.nn.l2_loss(gamma_neg)]) \
                           + self.l_b * tf.nn.l2_loss(beta_pos) \
                           + self.l_b * tf.nn.l2_loss(beta_neg) / 10

                # Regularized the loss to be optimized
                loss += reg_loss

            grad_F = tape_adv.gradient(loss, self.Delta_F)
            grad_F = tf.stop_gradient(grad_F)
            step_Delta_F = tf.nn.l2_normalize(grad_F, 1) * eps_iter

            self.Delta_F = tf.clip_by_value(self.Delta_F + step_Delta_F, -self.eps, self.eps)
            self.Delta_F = tf.Variable(self.Delta_F, trainable=True)
        return self.Delta_F
