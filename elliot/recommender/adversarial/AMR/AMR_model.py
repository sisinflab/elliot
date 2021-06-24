"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import numpy as np
import tensorflow as tf
from tensorflow import keras, Variable



class AMR_model(keras.Model):
    _Delta_Gu: Variable
    _Delta_Gi: Variable

    def __init__(self, factors=200, factors_d=20,
                 learning_rate=0.001,
                 l_w=0, l_b=0, l_e=0,
                 num_image_feature=2048,
                 num_users=100,
                 num_items=100,
                 eps=0.05,
                 l_adv=0.001,
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

        # Initialize the perturbation with 0 values
        self.Delta_Gu = tf.Variable(tf.zeros(shape=[self.num_users, self.factors]), dtype=tf.float32,
                                    trainable=False)
        self.Delta_Gi = tf.Variable(tf.zeros(shape=[self.num_items, self.factors]), dtype=tf.float32,
                                    trainable=False)

        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        #self.saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

    def call(self, inputs, training=None):
        user, item, feature_i = inputs
        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu + self.Delta_Gu, user))
        theta_u = tf.squeeze(tf.nn.embedding_lookup(self.Tu, user))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi + self.Delta_Gi, item))

        xui = beta_i + tf.reduce_sum((gamma_u * gamma_i), axis=1) + \
              tf.reduce_sum((theta_u * tf.matmul(feature_i, self.E)), axis=1) + \
              tf.squeeze(tf.matmul(feature_i, self.Bp))

        return xui, gamma_u, gamma_i, feature_i, theta_u, beta_i

    def train_step(self, batch, use_adv_train=False):
        user, pos, feature_pos, neg, feature_neg = batch
        with tf.GradientTape() as t:
            xu_pos, gamma_u, gamma_pos, _, theta_u, beta_pos = \
                self(inputs=(user, pos, feature_pos), training=True)
            xu_neg, _, gamma_neg, _, _, beta_neg = self(inputs=(user, neg, feature_neg), training=True)

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
                adv_xu_pos, _, _, _, _, _ = self(inputs=(user, pos, feature_pos), training=True)
                adv_xu_neg, _, _, _, _, _ = self(inputs=(user, neg, feature_neg), training=True)

                adv_difference = tf.clip_by_value(adv_xu_pos - adv_xu_neg, -80.0, 1e8)
                adv_loss = tf.reduce_sum(tf.nn.softplus(-adv_difference))

                loss += self.l_adv * adv_loss

        grads = t.gradient(loss, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp])
        self.optimizer.apply_gradients(zip(grads, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp]))

        return loss

    def predict(self, start, stop):
        return self.Bi + tf.matmul(self.Gu[start:stop], self.Gi, transpose_b=True) \
               + tf.matmul(self.Tu[start:stop], tf.matmul(self.F, self.E), transpose_b=True) \
               + tf.squeeze(tf.matmul(self.F, self.Bp))

    def predict_item_batch(self, start, stop, start_item, stop_item, feat):
        return self.Bi[start_item:(stop_item + 1)] + tf.matmul(self.Gu[start:stop], self.Gi[start_item:(stop_item + 1)],
                                                               transpose_b=True) \
               + tf.matmul(self.Tu[start:stop], tf.matmul(feat, self.E), transpose_b=True) \
               + tf.squeeze(tf.matmul(feat, self.Bp))

    def get_config(self):
        raise NotImplementedError

    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)

    def build_perturbation(self, batch):
        """
        Evaluate Adversarial Perturbation with FGSM-like Approach
        """
        self.Delta_Gu = self.Delta_Gu * 0.0
        self.Delta_Gi = self.Delta_Gi * 0.0

        user, pos, feature_pos, neg, feature_neg = batch
        with tf.GradientTape() as tape_adv:
            # Clean Inference
            xu_pos, gamma_u, gamma_pos, _, theta_u, beta_pos = \
                self(inputs=(user, pos, feature_pos), training=True)
            xu_neg, _, gamma_neg, _, _, beta_neg = self(inputs=(user, neg, feature_neg), training=True)

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

        grad_Gu, grad_Gi = tape_adv.gradient(loss, [self.Gu, self.Gi])
        grad_Gu, grad_Gi = tf.stop_gradient(grad_Gu), tf.stop_gradient(grad_Gi)
        self.Delta_Gu = tf.nn.l2_normalize(grad_Gu, 1) * self.eps
        self.Delta_Gi = tf.nn.l2_normalize(grad_Gi, 1) * self.eps
