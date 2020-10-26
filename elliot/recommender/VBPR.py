import logging
import os

import numpy as np
import tensorflow as tf

from dataset.visual_loader_mixin import VisualLoader
from recommender.BPRMF import BPRMF

np.random.seed(0)
tf.random.set_seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class VBPR(BPRMF, VisualLoader):

    def __init__(self, data, params):
        """
        Create a VBPR instance.
        (see https://arxiv.org/pdf/1510.01784.pdf for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w, l_b]: regularization,
                                      lr: learning rate}
        """
        super(VBPR, self).__init__(data, params)

        self.embed_k = self.params.embed_k
        self.embed_d = self.params.embed_d
        self.learning_rate = self.params.lr
        self.l_e = self.params.l_e

        self.process_visual_features(data)

        self.adv_eps = self.params.adv_eps

        # Initialize Model Parameters
        initializer = tf.initializers.GlorotUniform()
        self.Bp = tf.Variable(
            initializer(shape=[self.num_image_feature, 1]), name='Bp', dtype=tf.float32)
        self.Tu = tf.Variable(
            initializer(shape=[self.num_users, self.embed_d]),
            name='Tu', dtype=tf.float32)  # (users, low_embedding_size)
        self.F = tf.Variable(
            initializer(shape=[self.num_items, self.num_image_feature]),
            name='F', dtype=tf.float32, trainable=False)
        self.E = tf.Variable(
            initializer(shape=[self.num_image_feature, self.embed_d]),
            name='E', dtype=tf.float32)  # (items, low_embedding_size)

        self.set_delta()

        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)
        self.saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

    def set_delta(self, delta_init=0):
        """
        Set delta variables useful to store delta perturbations,
        :param delta_init: 0: zero-like initialization, 1 uniform random noise initialization
        :return:
        """
        if delta_init:
            self.delta_gu = tf.random.uniform(shape=[self.num_users, self.embed_k], minval=-0.05, maxval=0.05,
                                             dtype=tf.dtypes.float32, seed=0)
            self.delta_gi = tf.random.uniform(shape=[self.num_items, self.embed_k], minval=-0.05, maxval=0.05,
                                              dtype=tf.dtypes.float32, seed=0)
        else:
            self.delta_gu = tf.Variable(tf.zeros(shape=[self.num_users, self.embed_k]), dtype=tf.dtypes.float32,
                                       trainable=False)
            self.delta_gi = tf.Variable(tf.zeros(shape=[self.num_items, self.embed_k]), dtype=tf.dtypes.float32,
                                     trainable=False)

    def call(self, inputs, training=None, mask=None):
        """
        Generates prediction for passed users and items indices

        Args:
            inputs: user, item (batch)
            training: Boolean or boolean scalar tensor, indicating whether to run
            the `Network` in training mode or inference mode.
            mask: A mask or list of masks. A mask can be
            either a tensor or None (no mask).

        Returns:
            prediction and extracted model parameters
        """
        user, item = inputs
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        theta_u = tf.squeeze(tf.nn.embedding_lookup(self.Tu, user))

        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        feature_i = tf.squeeze(tf.nn.embedding_lookup(self.F, item))

        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))

        xui = beta_i + tf.reduce_sum(gamma_u * gamma_i, 1) + \
              tf.reduce_sum(theta_u * tf.matmul(feature_i, self.E), 1) + \
              tf.squeeze(tf.matmul(feature_i, self.Bp))

        return xui, gamma_u, gamma_i, feature_i, theta_u, beta_i

    def predict_all(self):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        return self.Bi + tf.matmul(self.Gu, self.Gi, transpose_b=True) \
               + tf.matmul(self.Tu, tf.matmul(self.F, self.E), transpose_b=True) \
               + tf.squeeze(tf.matmul(self.F, self.Bp))

    def train_step(self, batch):
        """
        Apply a single training step on one batch.

        Args:
            batch: batch used for the current train step

        Returns:
            loss value at the current batch
        """
        user, pos, neg = batch
        with tf.GradientTape() as t:

            # Clean Inference
            xu_pos, gamma_u, gamma_pos, emb_pos_feature, theta_u, beta_pos = \
                self(inputs=(user, pos), training=True)
            xu_neg, _, gamma_neg, _, _, beta_neg = self(inputs=(user, pos), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg),
                                                 tf.nn.l2_loss(theta_u)]) \
                    + self.l_b * tf.nn.l2_loss(beta_pos) \
                    + self.l_b * tf.nn.l2_loss(beta_pos)/10 \
                    + self.l_e * tf.nn.l2_loss(self.E, self.Bp)

            # Loss to be optimized
            loss += reg_loss

        grads = t.gradient(loss, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp])
        self.optimizer.apply_gradients(zip(grads, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp]))

        return loss.numpy()
