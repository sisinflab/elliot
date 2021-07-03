"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras, Variable


class MSAPMF_model(keras.Model):

    def __init__(self,
                 factors=200,
                 learning_rate=0.001,
                 l_w=0, l_b=0, eps=0.05, l_adv=0,
                 eps_iter=0.0005,
                 nb_iter=20,
                 num_users=100,
                 num_items=100,
                 random_seed=42,
                 name="MSAPMF",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self._factors = factors
        self._learning_rate = learning_rate
        self._l_w = l_w
        self._l_b = l_b
        self._l_adv = l_adv
        self._eps = eps
        self._eps_iter = eps_iter
        self._nb_iter = nb_iter
        self._num_items = num_items
        self._num_users = num_users

        self._initializer = tf.initializers.GlorotUniform()
        self._Bi = tf.Variable(tf.zeros(self._num_items), name='Bi', dtype=tf.float32)
        self._Gu = tf.Variable(self._initializer(shape=[self._num_users, self._factors]), name='Gu', dtype=tf.float32)
        self._Gi = tf.Variable(self._initializer(shape=[self._num_items, self._factors]), name='Gi', dtype=tf.float32)

        # Initialize the perturbation with 0 values
        self._Delta_Gu = tf.Variable(tf.zeros(shape=[self._num_users, self._factors]), dtype=tf.float32,
                                     trainable=False)
        self._Delta_Gi = tf.Variable(tf.zeros(shape=[self._num_items, self._factors]), dtype=tf.float32,
                                     trainable=False)

        self._optimizer = tf.optimizers.Adam(self._learning_rate)
        # self.saver_ckpt = tf.train.Checkpoint(optimizer=self._optimizer, model=self)

    # @tf.function
    def call(self, inputs, adversarial=False, training=None):
        user, item = inputs
        beta_i = tf.nn.embedding_lookup(self._Bi, item)
        if adversarial:
            gamma_u = tf.nn.embedding_lookup(self._Gu, user)
            gamma_i = tf.nn.embedding_lookup(self._Gi, item)
        else:
            gamma_u = tf.nn.embedding_lookup(self._Gu + self._Delta_Gu, user)
            gamma_i = tf.nn.embedding_lookup(self._Gi + self._Delta_Gi, item)

        xui = beta_i + tf.reduce_sum(gamma_u * gamma_i, 1)

        return xui, beta_i, gamma_u, gamma_i

    # @tf.function
    def train_step(self, batch, user_adv_train=False):
        user, pos, neg = batch
        with tf.GradientTape() as tape:
            # Clean Inference
            xu_pos, beta_pos, gamma_u, gamma_pos = self(inputs=(user, pos), adversarial=False, training=True)
            xu_neg, beta_neg, gamma_u, gamma_neg = self(inputs=(user, neg), adversarial=False, training=True)

            difference = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-difference))
            # Regularization Component
            reg_loss = self._l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                  tf.nn.l2_loss(gamma_pos),
                                                  tf.nn.l2_loss(gamma_neg)]) \
                       + self._l_b * tf.nn.l2_loss(beta_pos) \
                       + self._l_b * tf.nn.l2_loss(beta_neg) / 10

            # Loss to be optimized
            loss += reg_loss

            if user_adv_train:
                # Build the Adversarial Perturbation on the Current Model Parameters
                self.build_msap_perturbation(batch, self._eps_iter, self._nb_iter)

                # Clean Inference
                adv_xu_pos, _, _, _ = self(inputs=(user, pos), adversarial=True, training=True)
                adv_xu_neg, _, _, _ = self(inputs=(user, neg), adversarial=True, training=True)

                adv_difference = tf.clip_by_value(adv_xu_pos - adv_xu_neg, -80.0, 1e8)
                adv_loss = tf.reduce_sum(tf.nn.softplus(-adv_difference))

                loss += self._l_adv * adv_loss

        grads = tape.gradient(loss, [self._Bi, self._Gu, self._Gi])
        self._optimizer.apply_gradients(zip(grads, [self._Bi, self._Gu, self._Gi]))

        return loss

    # @tf.function
    def predict(self, start, stop, adversarial, **kwargs):
        if adversarial:
            return self._Bi + tf.matmul(self._Gu[start:stop] + self._Delta_Gu[start:stop],
                                        self._Gi + self._Delta_Gi, transpose_b=True)
        else:
            return self._Bi + tf.matmul(self._Gu[start:stop], self._Gi, transpose_b=True)

    # @tf.function
    def get_top_k(self, predictions, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, predictions, -np.inf), k=k, sorted=True)

    # @tf.function
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

    # @tf.function
    def build_perturbation(self, batch):
        """
        Evaluate Adversarial Perturbation with FGSM-like Approach
        """
        self._Delta_Gu = self._Delta_Gu * 0.0
        self._Delta_Gi = self._Delta_Gi * 0.0

        user, pos, neg = batch
        with tf.GradientTape() as tape_adv:
            # Clean Inference
            xu_pos, beta_pos, gamma_u, gamma_pos = self(inputs=(user, pos), training=True)
            xu_neg, beta_neg, gamma_u, gamma_neg = self(inputs=(user, neg), training=True)

            difference = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-difference))
            # Regularization Component
            reg_loss = self._l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                  tf.nn.l2_loss(gamma_pos),
                                                  tf.nn.l2_loss(gamma_neg)]) \
                       + self._l_b * tf.nn.l2_loss(beta_pos) \
                       + self._l_b * tf.nn.l2_loss(beta_neg) / 10

            # Loss to be optimized
            loss += reg_loss

        grad_Gu, grad_Gi = tape_adv.gradient(loss, [self._Gu, self._Gi])
        grad_Gu, grad_Gi = tf.stop_gradient(grad_Gu), tf.stop_gradient(grad_Gi)
        self._Delta_Gu = tf.nn.l2_normalize(grad_Gu, 1) * self._eps
        self._Delta_Gi = tf.nn.l2_normalize(grad_Gi, 1) * self._eps

    def build_msap_perturbation(self, batch, eps_iter, nb_iter):
        """
        Adversarial Perturbation with MSAP
        https://journals.flvc.org/FLAIRS/article/view/128443
        """
        self._Delta_Gu = self._Delta_Gu * 0.0
        self._Delta_Gi = self._Delta_Gi * 0.0

        for _ in range(nb_iter):
            user, pos, neg = batch
            with tf.GradientTape() as tape_adv:
                # Clean Inference
                xu_pos, beta_pos, gamma_u, gamma_pos = self(inputs=(user, pos), training=True)
                xu_neg, beta_neg, gamma_u, gamma_neg = self(inputs=(user, neg), training=True)

                difference = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
                loss = tf.reduce_sum(tf.nn.softplus(-difference))
                # Regularization Component
                reg_loss = self._l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                      tf.nn.l2_loss(gamma_pos),
                                                      tf.nn.l2_loss(gamma_neg)]) \
                           + self._l_b * tf.nn.l2_loss(beta_pos) \
                           + self._l_b * tf.nn.l2_loss(beta_neg) / 10

                # Regularized the loss to be optimized
                loss += reg_loss

            grad_Gu, grad_Gi = tape_adv.gradient(loss, [self._Gu, self._Gi])
            grad_Gu, grad_Gi = tf.stop_gradient(grad_Gu), tf.stop_gradient(grad_Gi)

            step_Delta_Gu = tf.nn.l2_normalize(grad_Gu, 1) * eps_iter
            step_Delta_Gi = tf.nn.l2_normalize(grad_Gi, 1) * eps_iter

            self._Delta_Gu = tf.clip_by_value(self._Delta_Gu + step_Delta_Gu, -self._eps, self._eps)
            self._Delta_Gi = tf.clip_by_value(self._Delta_Gi + step_Delta_Gi, -self._eps, self._eps)
