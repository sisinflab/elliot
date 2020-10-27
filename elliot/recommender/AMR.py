import logging
import os

import numpy as np
import tensorflow as tf

from recommender.VBPR import VBPR

np.random.seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AMR(VBPR):

    def __init__(self, data, params):
        super(AMR, self).__init__(data, params)
        self.adv_type = self.params.adv_type
        self.adv_reg = self.params.adv_reg
        self.set_delta(delta_init=params.delta)

    def call(self, inputs, training=None, mask=None):
        user, item = inputs
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        theta_u = tf.squeeze(tf.nn.embedding_lookup(self.Tu, user))

        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        feature_i = tf.squeeze(tf.nn.embedding_lookup(self.F + self.delta, item))

        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))

        xui = beta_i + tf.reduce_sum(gamma_u * gamma_i, 1) + \
              tf.reduce_sum(theta_u * tf.matmul(feature_i, self.E), 1) + \
              tf.squeeze(tf.matmul(feature_i, self.Bp))

        return xui, gamma_u, gamma_i, feature_i, theta_u, beta_i

    def predict_all_adv(self):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        if self.adv_type == 'fgsm':
            user, pos, neg = self.data.shuffle(len(self.data._user_input))
            self.fgsm_perturbation(user, pos, neg)
        else:
            self.rand_perturbation()
        return self.Bi + tf.matmul(self.Gu, self.Gi, transpose_b=True) \
               + tf.matmul(self.Tu, tf.matmul((self.F + self.delta), self.E), transpose_b=True) \
               + tf.squeeze(tf.matmul(self.F, self.Bp))

    def train_step(self, batch):
        """
                Apply a single training step (across all batched in the dataset).
                :param batches: set of batches used fr the training
                :return:
                """
        user, pos, neg = batch
        self.set_delta(delta_init=0)
        loss_adver = 0

        with tf.GradientTape() as t:

            t.watch([self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp])

            # Clean Inference
            xu_pos, gamma_u, gamma_pos, emb_pos_feature, theta_u, beta_pos = \
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
                    + self.l_b * tf.nn.l2_loss(beta_neg)/10 \
                    + self.l_e * tf.reduce_sum([tf.nn.l2_loss(self.E), tf.nn.l2_loss(self.Bp)])

            if self.epoch >= 2000:
                if self.adv_type == 'fgsm':
                    self.fgsm_perturbation(user, pos, neg)
                elif self.adv_type == 'rand':
                    self.rand_perturbation()

                xu_pos,  _, _, _, _, _ = self(inputs=(user, pos), training=True)
                xu_neg, _, _, _, _, _ = self(inputs=(user, neg), training=True)
                result_adver = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
                loss_adver = tf.reduce_sum(tf.nn.softplus(-result_adver))

            loss += self.adv_reg * loss_adver + reg_loss

        gradients = t.gradient(self.loss_opt, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp])
        self.optimizer.apply_gradients(zip(gradients, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp]))

        return loss.numpy()

    def fgsm_perturbation(self, user, pos, neg, ):
        """
        Evaluate Adversarial Perturbation with FGSM-like Approach
        :param user_input:
        :param item_input_pos:
        :param item_input_neg:
        :param batch_idx:
        :return:
        """
        with tf.GradientTape() as tape_adv:
            tape_adv.watch(self.F)
            # Clean Inference
            pos_pred, gamma_u, gamma_pos, emb_pos_feature, theta_u, beta_pos = \
                self(inputs=(user, pos), training=True)
            neg_pred, _, gamma_neg, _, _, beta_neg = self(inputs=(user, neg), training=True)
            result = tf.clip_by_value(pos_pred - neg_pred, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))
            loss += self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg),
                                                 tf.nn.l2_loss(theta_u)]) \
                    + self.l_b * tf.nn.l2_loss(beta_pos) \
                    + self.l_b * tf.nn.l2_loss(beta_neg)/10 \
                    + self.l_e * tf.nn.l2_loss(self.E, self.Bp)

        d = tape_adv.gradient(loss, [self.F])[0]
        d = tf.stop_gradient(d)
        self.delta = self.adv_eps * tf.nn.l2_normalize(d, 1)

    def rand_perturbation(self):
        initializer = tf.initializers.GlorotUniform()
        d = tf.Variable(
                initializer(shape=self.F.shape), name='delta', dtype=tf.float32)
        self.delta = d

