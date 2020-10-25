import logging
import os

import numpy as np
import tensorflow as tf

from dataset.visual_loader_mixin import VisualLoader
from recommender.BPRMF_new import BPRMF
from recommender.Evaluator import Evaluator

np.random.seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class VBPR(BPRMF, VisualLoader):

    def __init__(self, data, params):
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
            initializer(shape=[self.embed_d, self.num_image_feature]),
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
        generate predicition matrix with respect to passed users' and items indices
        :param user_input: user indices
        :param item_input_pos: item indices
        :return:
        """
        user, item = inputs
        gamma_u = tf.nn.embedding_lookup(self.Gu, user)
        theta_u = tf.nn.embedding_lookup(self.Tu, user)

        gamma_i = tf.nn.embedding_lookup(self.Gi, item)
        feature_i = tf.nn.embedding_lookup(self.F, item)

        beta_i = tf.nn.embedding_lookup(self.Bi, item)

        xui = beta_i + tf.tensordot(gamma_u, gamma_i,  axes=[(1, 2), (1, 2)]) + \
              tf.tensordot(theta_u, tf.tensordot(self.E, feature_i, axes=[[1], [2]]), axes=[(2, 1), (0, 2)]) + \
              tf.tensordot(feature_i, self.Bp, axes=[(1, 2), (0, 1)])

        return xui, gamma_u, gamma_i, feature_i, theta_u, beta_i

    def predict_all(self):
        """
        Get Full Predictions useful for Full Store of Predictions
        :return: The matrix of predicted values.
        """
        return self.Bi + tf.tensordot(self.Gu, self.Gi, axes=[[1], [1]]) \
               + tf.tensordot(self.Tu, tf.matmul(self.F, self.E), axes=[[1], [1]]) \
               + tf.matmul(self.F, self.Bp)

    def train_step(self, batches):
        """
                Apply a single training step (across all batched in the dataset).
                :param batches: set of batches used fr the training
                :return:

                """
        for user, pos, neg in zip(*batches):
            with tf.GradientTape() as t:
                t.watch([self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp])

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