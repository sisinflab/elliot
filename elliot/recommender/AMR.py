import numpy as np
from time import time
from recommender.Evaluator import Evaluator
import os
import logging

from recommender.VBPR import VBPR
from utils.write import save_obj
from utils.read import find_checkpoint
from copy import deepcopy
import tensorflow as tf
from recommender.RecommenderModel import RecommenderModel


np.random.seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AMR(VBPR):

    def __init__(self, data, path_output_rec_result, path_output_rec_weight, args):
        super(AMR, self).__init__(data, path_output_rec_result, path_output_rec_weight, args.rec)
        self.adv_type = args.adv_type
        self.adv_reg = args.adv_reg

    def set_delta(self, delta_init=0):
        """
        Set delta variables useful to store delta perturbations,
        :param delta_init: 0: zero-like initialization, 1 uniform random noise initialization
        :return:
        """
        if delta_init:
            self.delta = tf.random.uniform(shape=self.F.shape, minval=-0.05, maxval=0.05,
                                              dtype=tf.dtypes.float32, seed=0)
        else:
            self.delta = tf.Variable(tf.zeros(shape=self.F.shape), dtype=tf.dtypes.float32,
                                        trainable=False)

    def _train_step(self, batches):
        """
                Apply a single training step (across all batched in the dataset).
                :param batches: set of batches used fr the training
                :return:
                """
        user_input, item_input_pos, item_input_neg = batches

        for batch_idx in range(len(user_input)):

            self.set_delta(delta_init=0)

            with tf.GradientTape() as t:

                t.watch([self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp])

                # Clean Inference
                self.pos_pred, gamma_u, gamma_i_pos, self.emb_pos_feature, theta_u, beta_i_pos = self.get_inference(
                    user_input[batch_idx],
                    item_input_pos[batch_idx])
                self.neg_pred, _, gamma_i_neg, _, _, beta_i_neg = self.get_inference(user_input[batch_idx],
                                                                               item_input_neg[batch_idx])

                self.result = tf.clip_by_value(self.pos_pred - self.neg_pred, -80.0, 1e8)
                self.loss = tf.reduce_sum(tf.nn.softplus(-self.result))

                # Regularization Component
                self.reg_loss = self.l_w * self._l2_loss(gamma_u, gamma_i_pos, gamma_i_neg, theta_u) \
                        + self.l_b * self._l2_loss(beta_i_pos) \
                        + self.l_b * self._l2_loss(beta_i_neg)/10 \
                        + self.l_e * self._l2_loss(self.E, self.Bp)

                if self.adv_type == 'fgsm':
                    self.fgsm_perturbation(user_input, item_input_pos, item_input_neg, batch_idx)
                elif self.adv_type == 'rand':
                    self.rand_perturbation()

                self.pos_pred,  _, _, _, _, _ = self.get_inference(
                    user_input[batch_idx],
                    item_input_pos[batch_idx])
                self.neg_pred, _, _, _, _, _ = self.get_inference(user_input[batch_idx], item_input_neg[batch_idx])
                self.result_adver = tf.clip_by_value(self.pos_pred - self.neg_pred, -80.0, 1e8)
                self.loss_adver = tf.reduce_sum(tf.nn.softplus(-self.result_adver))

                self.loss_opt = self.loss + self.adv_reg * self.loss_adver + self.reg_loss

            gradients = t.gradient(self.loss_opt, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp])
            self.optimizer.apply_gradients(zip(gradients, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp]))

    def fgsm_perturbation(self, user_input, item_input_pos, item_input_neg, batch_idx=0):
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
            pos_pred, gamma_u, gamma_i_pos, emb_pos_feature, theta_u, beta_i_pos = self._forward(
                user_input[batch_idx],
                item_input_pos[batch_idx])
            neg_pred, _, gamma_i_neg, _, _, beta_i_neg = self._forward(user_input[batch_idx],
                                                                            item_input_neg[batch_idx])
            result = tf.clip_by_value(pos_pred - neg_pred, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))
            loss += self.l_w * self._l2_loss(gamma_u, gamma_i_pos, gamma_i_neg, theta_u) \
                        + self.l_b * self._l2_loss(beta_i_pos) \
                        + self.l_b * self._l2_loss(beta_i_neg)/10 \
                        + self.l_e * self._l2_loss(self.E, self.Bp)

        d = tape_adv.gradient(loss, [self.F])[0]
        d = tf.stop_gradient(d)
        self.feature_i = self.adv_eps * tf.nn.l2_normalize(d, 1)

    def random_perturbation(self):
        initializer = tf.initializers.GlorotUniform()
        d = tf.Variable(
                initializer(shape=self.F.shape), name='delta', dtype=tf.float32)
        self.feature_i += d
