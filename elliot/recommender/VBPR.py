import numpy as np
from time import time

from dataset.visual_loader_mixin import VisualLoader
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


class VBPR(RecommenderModel, VisualLoader):

    def __init__(self, data, path_output_rec_result, path_output_rec_weight, args):
        super(VBPR, self).__init__(data, path_output_rec_result, path_output_rec_weight, args.rec)

        self.emb_K = args.emb1_K
        self.emb_D = args.emb1_D
        self.learning_rate = args.lr
        self.l_w = args.reg_w
        self.l_b = args.reg_b
        self.l_e = args.reg_e
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.verbose = args.verbose
        self.restore_epochs = args.restore_epochs
        self.epsilon = args.epsilon
        self.process_visual_features(data)
        self.evaluator = Evaluator(self, data, args.k)
        self.best = args.best

        # Initialize Model Parameters
        initializer = tf.initializers.GlorotUniform()
        self.Bi = tf.Variable(
            tf.zeros(self.num_items), name='Bi', dtype=tf.float32)  # (items, 1)
        self.Bp = tf.Variable(
            initializer(shape=[self.num_image_feature, 1]), name='Bp', dtype=tf.float32)
        self.Gu = tf.Variable(
            initializer(shape=[self.num_users, self.emb_K]),
            name='Gu', dtype=tf.float32)  # (users, embedding_size)
        self.Gi = tf.Variable(
            initializer(shape=[self.num_items, self.emb_K]),
            name='Gi', dtype=tf.float32)  # (items, embedding_size)
        self.Tu = tf.Variable(
            initializer(shape=[self.num_users, self.emb_D]),
            name='Tu', dtype=tf.float32)  # (users, low_embedding_size)
        self.F = tf.Variable(
            initializer(shape=[self.num_items, self.num_image_feature]),
            name='F', dtype=tf.float32, trainable=False)
        self.E = tf.Variable(
            initializer(shape=[self.num_image_feature, self.emb_D]),
            name='E', dtype=tf.float32)  # (items, low_embedding_size)

        self.set_delta()

        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)

    def set_delta(self, delta_init=0):
        """
        Set delta variables useful to store delta perturbations,
        :param delta_init: 0: zero-like initialization, 1 uniform random noise initialization
        :return:
        """
        if delta_init:
            self.delta_gu = tf.random.uniform(shape=[self.num_users, self.emb_K], minval=-0.05, maxval=0.05,
                                             dtype=tf.dtypes.float32, seed=0)
            self.delta_gi = tf.random.uniform(shape=[self.num_items, self.emb_K], minval=-0.05, maxval=0.05,
                                              dtype=tf.dtypes.float32, seed=0)
        else:
            self.delta_gu = tf.Variable(tf.zeros(shape=[self.num_users, self.emb_K]), dtype=tf.dtypes.float32,
                                       trainable=False)
            self.delta_gi = tf.Variable(tf.zeros(shape=[self.num_items, self.emb_K]), dtype=tf.dtypes.float32,
                                     trainable=False)

    def get_inference(self, user_input, item_input_pos):
        """
        generate predicition matrix with respect to passed users' and items indices
        :param user_input: user indices
        :param item_input_pos: item indices
        :return:
        """
        self.gamma_u = tf.nn.embedding_lookup(self.Gu, user_input)
        self.theta_u = tf.nn.embedding_lookup(self.Tu, user_input)

        self.gamma_i = tf.nn.embedding_lookup(self.Gi, item_input_pos)
        self.feature_i = tf.nn.embedding_lookup(self.F, item_input_pos)

        self.beta_i = tf.nn.embedding_lookup(self.Bi, item_input_pos)

        xui = self.beta_i + tf.reduce_sum((self.gamma_u * self.gamma_i), axis=1) + \
              tf.reduce_sum((self.theta_u * tf.matmul(self.feature_i, self.E)), axis=1) + \
              tf.matmul(self.feature_i, self.Bp)

        return xui, self.gamma_u, self.gamma_i, self.feature_i, self.theta_u, self.beta_i

    def get_full_inference(self):
        """
        Get Full Predictions useful for Full Store of Predictions
        :return: The matrix of predicted values.
        """
        return self.Bi + tf.tensordot(self.Gu, self.Gi, axes=[[1], [1]]) \
               + tf.tensordot(self.Tu, tf.matmul(self.F, self.E), axes=[[1], [1]]) \
               + tf.matmul(self.F, self.Bp)

    def _train_step(self, batches):
        """
                Apply a single training step (across all batched in the dataset).
                :param batches: set of batches used fr the training
                :return:
                """
        user_input, item_input_pos, item_input_neg = batches

        for batch_idx in range(len(user_input)):
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

                # Loss to be optimized
                self.loss_opt = self.loss + self.reg_loss

            gradients = t.gradient(self.loss_opt, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp])
            self.optimizer.apply_gradients(zip(gradients, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp]))

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