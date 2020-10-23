"""
Created on April 4, 2020
Tensorflow 2.1.0 implementation of BPR-MF.
@author Anonymized
"""
import numpy as np
from time import time
from recommender.Evaluator import Evaluator
import os
import logging
from utils.write import save_obj
from utils.read import find_checkpoint
from copy import deepcopy
import tensorflow as tf
from recommender.RecommenderModel import RecommenderModel


np.random.seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BPRMF(RecommenderModel):

    def __init__(self, data, path_output_rec_result, path_output_rec_weight, args):
        """
        Create a BPR-MF instance.
        (see https://arxiv.org/pdf/1205.2618 for details about the algorithm design choices)
        :param data: data loader object
        :param path_output_rec_result: path to the directory rec. results
        :param path_output_rec_weight: path to the directory rec. model parameters
        :param args: parameters
        """
        super(BPRMF, self).__init__(data, path_output_rec_result, path_output_rec_weight, args.rec)
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        self.reg = args.reg
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.verbose = args.verbose
        self.restore_epochs = args.restore_epochs
        self.evaluator = Evaluator(self, data, args.k)
        self.best = args.best

        # Initialize Model Parameters
        self.embedding_P = tf.Variable(
            tf.random.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
            name='embedding_P', dtype=tf.dtypes.float32)  # (users, embedding_size)
        self.embedding_Q = tf.Variable(
            tf.random.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
            name='embedding_Q', dtype=tf.dtypes.float32)  # (items, embedding_size)
        self.h = tf.constant(1.0, tf.float32, [self.embedding_size, 1], name="h")

        # Method to create delta locations for the adv. perturbations
        self.set_delta()

        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)

    def set_delta(self, delta_init=0):
        """
        Set delta variables useful to store delta perturbations,
        :param delta_init: 0: zero-like initialization, 1 uniform random noise initialization
        :return:
        """
        if delta_init:
            self.delta_P = tf.random.uniform(shape=[self.num_users, self.embedding_size], minval=-0.05, maxval=0.05,
                                             dtype=tf.dtypes.float32, seed=0)
            self.delta_Q = tf.random.uniform(shape=[self.num_items, self.embedding_size], minval=-0.05, maxval=0.05,
                                             dtype=tf.dtypes.float32, seed=0)
        else:
            self.delta_P = tf.Variable(tf.zeros(shape=[self.num_users, self.embedding_size]), dtype=tf.dtypes.float32,
                                       trainable=False)
            self.delta_Q = tf.Variable(tf.zeros(shape=[self.num_items, self.embedding_size]), dtype=tf.dtypes.float32,
                                       trainable=False)

    def get_inference(self, user_input, item_input_pos):
        """
        generate predicition matrix with respect to passed users' and items indices
        :param user_input: user indices
        :param item_input_pos: item indices
        :return:
        """
        self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P + self.delta_P, user_input), 1)
        self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q + self.delta_Q, item_input_pos), 1)

        return tf.matmul(self.embedding_p * self.embedding_q,
                         self.h), self.embedding_p, self.embedding_q  # (b, embedding_size) * (embedding_size, 1)

    def get_full_inference(self):
        """
        Get Full Predictions useful for Full Store of Predictions
        :return: The matrix of predicted values.
        """
        return tf.matmul(self.embedding_P + self.delta_P, tf.transpose(self.embedding_Q + self.delta_Q))

    def _train_step(self, batches):
        """
        Apply a single training step (across all batched in the dataset).
        :param batches: set of batches used fr the training
        :return:
        """
        user_input, item_input_pos, item_input_neg = batches

        for batch_idx in range(len(user_input)):
            with tf.GradientTape() as t:
                t.watch([self.embedding_P, self.embedding_Q])

                # Clean Inference
                self.output_pos, embed_p_pos, embed_q_pos = self.get_inference(user_input[batch_idx],
                                                                               item_input_pos[batch_idx])
                self.output_neg, embed_p_neg, embed_q_neg = self.get_inference(user_input[batch_idx],
                                                                               item_input_neg[batch_idx])
                self.result = tf.clip_by_value(self.output_pos - self.output_neg, -80.0, 1e8)
                self.loss = tf.reduce_sum(tf.nn.softplus(-self.result))

                # Regularization Component
                self.reg_loss = self.reg * tf.reduce_mean(
                    tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg))

                # Loss to be optimized
                self.loss_opt = self.loss + self.reg_loss

            gradients = t.gradient(self.loss_opt, [self.embedding_P, self.embedding_Q])
            self.optimizer.apply_gradients(zip(gradients, [self.embedding_P, self.embedding_Q]))

    def train(self):

        saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

        if self.restore():
            self.restore_epochs += 1
        else:
            self.restore_epochs = 1
            print("Training from scratch...")

        # initialize the max_ndcg to memorize the best result
        max_hr = 0
        best_model = self
        best_epoch = self.restore_epochs
        results = {}

        for epoch in range(self.restore_epochs, self.epochs + 1):
            startep = time()
            batches = self.data.shuffle(self.batch_size)
            self._train_step(batches)
            epoch_text = 'Epoch {0}/{1}'.format(epoch, self.epochs)
            self.evaluator.eval(epoch, results, epoch_text, startep)

            # print and log the best result (HR@10)
            if max_hr < results[epoch]['hr'][self.evaluator.k - 1]:
                max_hr = results[epoch]['hr'][self.evaluator.k - 1]
                best_epoch = epoch
                best_model = deepcopy(self)

            if epoch % self.verbose == 0 or epoch == 1:
                saver_ckpt.save('{0}/weights-{1}'.format(self.path_output_rec_weight, epoch))

        self.evaluator.store_recommendation()
        save_obj(results,
                 '{0}/{1}-results'.format(self.path_output_rec_result, self.path_output_rec_result.split('/')[-2]))

        # Store the best model
        print("Store Best Model at Epoch {0}".format(best_epoch))
        saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=best_model)
        saver_ckpt.save('{0}/best-weights-{1}'.format(self.path_output_rec_weight, best_epoch))
        best_model.evaluator.store_recommendation()

    def restore(self):
        saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
        if self.best:
            try:
                checkpoint_file = find_checkpoint(self.path_output_rec_weight, 0, 0, self.rec, self.best)
                saver_ckpt.restore(checkpoint_file)
                print("Best Model correctly Restored: {0}".format(checkpoint_file))
                return True
            except Exception as ex:
                print("Error in model restoring operation! {0}".format(ex.message))
                return False

        if self.restore_epochs > 1:
            try:
                checkpoint_file = find_checkpoint(self.path_output_rec_weight, self.restore_epochs, self.epochs,
                                                  self.rec)
                saver_ckpt.restore(checkpoint_file)
                print("Model correctly Restored at Epoch: {0}".format(self.restore_epochs))
                return True
            except Exception as ex:
                print("Error in model restoring operation! {0}".format(ex.message))
        else:
            print("Restore Epochs Not Specified")
        return False

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
            tape_adv.watch([self.embedding_P, self.embedding_Q])
            # Clean Inference
            output_pos, embed_p_pos, embed_q_pos = self.get_inference(user_input[batch_idx],
                                                                      item_input_pos[batch_idx])
            output_neg, embed_p_neg, embed_q_neg = self.get_inference(user_input[batch_idx],
                                                                      item_input_neg[batch_idx])
            result = tf.clip_by_value(output_pos - output_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))
            loss += self.reg * tf.reduce_mean(
                tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg))

        grad_P, grad_Q = tape_adv.gradient(loss, [self.embedding_P, self.embedding_Q])
        grad_P, grad_Q = tf.stop_gradient(grad_P), tf.stop_gradient(grad_Q)
        self.delta_P = tf.nn.l2_normalize(grad_P, 1) * self.eps
        self.delta_Q = tf.nn.l2_normalize(grad_Q, 1) * self.eps

    def iterative_perturbation(self, user_input, item_input_pos, item_input_neg, batch_idx=0):
        """
        Evaluate Adversarial Perturbation with Iterative-like Approach
        :param user_input:
        :param item_input_pos:
        :param item_input_neg:
        :param batch_idx:
        :return:
        """
        for _ in range(self.iteration):
            with tf.GradientTape() as tape_adv:
                tape_adv.watch([self.embedding_P, self.embedding_Q])
                # Clean Inference
                output_pos, embed_p_pos, embed_q_pos = self.get_inference(user_input[batch_idx],
                                                                          item_input_pos[batch_idx])
                output_neg, embed_p_neg, embed_q_neg = self.get_inference(user_input[batch_idx],
                                                                          item_input_neg[batch_idx])
                result = tf.clip_by_value(output_pos - output_neg, -80.0, 1e8)
                loss = tf.reduce_sum(tf.nn.softplus(-result))
                loss += self.reg * tf.reduce_mean(
                    tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg))

            grad_P, grad_Q = tape_adv.gradient(loss, [self.embedding_P, self.embedding_Q])
            grad_P, grad_Q = tf.stop_gradient(grad_P), tf.stop_gradient(grad_Q)
            self.step_delta_P = tf.nn.l2_normalize(grad_P, 1) * self.step_size
            self.step_delta_Q = tf.nn.l2_normalize(grad_Q, 1) * self.step_size

            # Clipping perturbation eta to norm norm ball
            # eta = adv_x - x
            # eta = clip_eta(eta, norm, eps)
            # adv_x = x + eta

            # L2 NORM on P
            # self.norm_P = tf.sqrt(tf.maximum(1e-12, tf.reduce_sum(tf.square(self.step_delta_P),
            #                                                       list(range(1, len(self.step_delta_P.get_shape()))),
            #                                                       keepdims=True)))
            # # We must *clip* to within the norm ball, not *normalize* onto the surface of the ball
            # self.factor_P = tf.minimum(1., tf.divide(self.eps, self.norm_P))
            #
            # # L2 NORM on Q
            # self.norm_Q = tf.sqrt(tf.maximum(1e-12, tf.reduce_sum(tf.square(self.step_delta_Q),
            #                                                       list(range(1, len(self.step_delta_Q.get_shape()))),
            #                                                       keepdims=True)))
            # self.factor_Q = tf.minimum(1., tf.divide(self.eps, self.norm_Q))
            #
            # self.delta_P = self.delta_P + self.step_delta_P * self.factor_P
            # self.delta_Q = self.delta_Q + self.step_delta_Q * self.factor_Q

            self.delta_P = tf.clip_by_value(self.delta_P + self.step_delta_P, -self.adv_eps, self.adv_eps)
            self.delta_Q = tf.clip_by_value(self.delta_Q + self.step_delta_Q, -self.adv_eps, self.adv_eps)

            if np.any(self.delta_P > self.adv_eps) or np.any(self.delta_Q > self.adv_eps):
                print('Test Pert.\nP is out the clip? {0}\nQ is out the clip? {1} \n- MAX P {2} - Q {3}'.format(
                    np.any(self.delta_P > self.adv_eps), np.any(self.delta_Q > self.adv_eps),
                    np.max(self.delta_P), np.max(self.delta_Q)))

    def attack_full_fgsm(self, attack_eps, attack_name=""):
        """
        Create FGSM ATTACK
        :param attack_eps:
        :param attack_name:
        :return:
        """
        # Set eps perturbation (budget)
        self.eps = attack_eps
        user_input, item_input_pos, item_input_neg = self.data.shuffle(len(self.data._user_input))
        print('Initial Performance.')
        self.evaluator.eval(self.restore_epochs, {}, 'BEST MODEL ' if self.best else str(self.restore_epochs))

        # Calculate Adversarial Perturbations
        self.fgsm_perturbation(user_input, item_input_pos, item_input_neg)

        results = {}
        print('After Attack Performance.')
        self.evaluator.eval(self.restore_epochs, results, 'BEST MODEL ' if self.best else str(self.restore_epochs))
        self.evaluator.store_recommendation(attack_name=attack_name)
        save_obj(results, '{0}/{1}-results'.format(self.path_output_rec_result,
                                                   attack_name + self.path_output_rec_result.split('/')[
                                                       -2] + '_best{0}'.format(self.best)))

        print('{0} - Completed!'.format(attack_name))

    def attack_full_iterative(self, attack_type, attack_iteration, attack_eps, attack_step_size, initial=1,
                              attack_name=""):
        """
        ITERATIVE ATTACKS (BIM and PGD)
        Inspired by compuer vision attacks:
        BIM: Kurakin et al. http://arxiv.org/abs/1607.02533
        PGD: Madry et al. https://arxiv.org/pdf/1706.06083.pdf
        :param attack_type: BIM/PGD
        :param attack_iteration: number of iterations
        :param attack_eps: clipping perturbation
        :param attack_step_size: step size perturbation
        :param attack_name: attack_name to be printed in the results
        :return:
        """
        # Set Iterative Parameters
        self.adv_eps = attack_eps
        self.step_size = attack_eps / attack_step_size
        self.iteration = attack_iteration

        if attack_type == 'pgd':
            self.set_delta(delta_init=1)
        else:
            self.set_delta(delta_init=0)

        user_input, item_input_pos, item_input_neg = self.data.shuffle(len(self.data._user_input))
        if initial:
            print('Initial Performance.')
            self.evaluator.eval(self.restore_epochs, {}, 'BEST MODEL ' if self.best else str(self.restore_epochs))
            self.evaluator.store_recommendation(attack_name='')
        # Calculate Adversarial Perturbations
        self.iterative_perturbation(user_input, item_input_pos, item_input_neg)

        results = {}
        print('After Attack Performance.')
        self.evaluator.eval(self.restore_epochs, results, 'BEST MODEL ' if self.best else str(self.restore_epochs))
        self.evaluator.store_recommendation(attack_name='/' + attack_name)
        save_obj(results, '{0}/{1}-results'.format(self.path_output_rec_result,
                                                   attack_name + self.path_output_rec_result.split('/')[
                                                       -2] + '_best{0}'.format(self.best)))

        print('{0} - Completed!'.format(attack_name))
