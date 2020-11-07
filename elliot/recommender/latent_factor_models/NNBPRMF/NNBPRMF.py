import logging
import os
from copy import deepcopy
from time import time

import numpy as np
import tensorflow as tf

from config.configs import *
from dataset.dataset import DataSet
from dataset.samplers import custom_sampler as cs
from evaluation.evaluator import Evaluator
from recommender.keras_base_recommender_model import RecommenderModel
from utils.read import find_checkpoint
from utils.write import save_obj, store_recommendation

np.random.seed(0)
tf.random.set_seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BPRMF(RecommenderModel):

    def __init__(self, config, params, *args, **kwargs):
        """
        Create a BPR-MF instance.
        (see https://arxiv.org/pdf/1205.2618 for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w, l_b]: regularization,
                                      lr: learning rate}
        """
        super().__init__(config, params, *args, **kwargs)
        np.random.seed(42)

        self._data = DataSet(config, params)
        self._config = config
        self._params = params
        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random
        self._sample_negative_items_empirically = True
        self._num_iters = self.params.epochs
        self._factors = self.params.embed_k
        self._learning_rate = self.params.lr
        self.l_w = self.params.l_w
        self.l_b = self.params.l_b

        self._restore_epochs = 0

        self._ratings = self._data.train_dataframe_dict

        self._sampler = cs.Sampler(self._ratings, self._random, self._sample_negative_items_empirically)

        self._iteration = 0

        self.evaluator = Evaluator(self._data)
        self._results = []


        # Initialize Model Parameters
        self.initializer = tf.initializers.GlorotUniform()
        self.Bi = tf.Variable(tf.zeros(self._num_items), name='Bi', dtype=tf.float32)
        self.Gu = tf.Variable(self.initializer(shape=[self._num_users, self._factors]), name='Gu', dtype=tf.float32)
        self.Gi = tf.Variable(self.initializer(shape=[self._num_items, self._factors]), name='Gi', dtype=tf.float32)

        self.optimizer = tf.optimizers.Adam(self._learning_rate)
        self.saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

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
        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))

        xui = beta_i + tf.reduce_sum(gamma_u * gamma_i, 1)

        return xui, beta_i, gamma_u, gamma_i

    def predict_all(self):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        return self.Bi + tf.matmul(self.Gu, self.Gi, transpose_b=True)

    def one_epoch(self, batches):
        """
        #TODO comment
        Args:
            batches:

        Returns:

        """
        loss = 0
        steps = 0
        for batch in zip(*batches):
            steps += 1
            loss += self.train_step(batch)
        return loss/steps

    def train_step(self, batch):
        """
        Apply a single training step on one batch.

        Args:
            batch: batch used for the current train step

        Returns:
            loss value at the current batch
        """
        user, pos, neg = batch
        with tf.GradientTape() as tape:

            # Clean Inference
            xu_pos, beta_pos, gamma_u, gamma_pos = self.call(inputs=(user, pos), training=True)
            xu_neg, beta_neg, gamma_u, gamma_neg = self.call(inputs=(user, neg), training=True)

            difference = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-difference))
            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg)]) \
                       + self.l_b * tf.nn.l2_loss(beta_pos) \
                       + self.l_b * tf.nn.l2_loss(beta_neg)/10

            # Loss to be optimized
            loss += reg_loss

        grads = tape.gradient(loss, [self.Bi, self.Gu, self.Gi])
        self.optimizer.apply_gradients(zip(grads, [self.Bi, self.Gu, self.Gi]))

        return loss.numpy()

    def train(self):
        if self.restore():
            self._restore_epochs += 1
        else:
            print("Training from scratch...")

        for it in range(self._restore_epochs, self._num_iters + 1):
            batches = self._sampler.step(self._data.transactions, self.params.batch_size)
            loss = self.one_epoch(batches)

            recs = self.get_recommendations(self._config.top_k)
            self._results.append(self.evaluator.eval(recs))

            print('Epoch {0}/{1} loss {2:.3f}'.format(it, self._num_iters, loss))

            if not (it + 1) % 10:
                store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}_{it + 1}.tsv")

    def restore(self):
        if self._restore_epochs > 1:
            try:
                checkpoint_file = find_checkpoint(weight_dir, self.restore_epochs, self.epochs,
                                                  self.rec)
                self.saver_ckpt.restore(checkpoint_file)
                print("Model correctly Restored at Epoch: {0}".format(self.restore_epochs))
                return True
            except Exception as ex:
                print("Error in model restoring operation! {0}".format(ex))
        else:
            print("Restore Epochs Not Specified")
        return False

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        for index, offset in enumerate(range(0, self.Gu.shape[0], self.params.batch_size)):
            predictions = (self.Bi + tf.matmul(self.Gu[offset: offset + self.params.batch_size],
                                               self.Gi, transpose_b=True))
            v, i = tf.nn.top_k(predictions, k=k, sorted=True)
            items_ratings_pair = [list(zip(map(self._sampler.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset + self.params.batch_size), items_ratings_pair)))
        return predictions_top_k


    def get_loss(self):
        pass

    def get_params(self):
        pass

    def get_results(self):
        pass