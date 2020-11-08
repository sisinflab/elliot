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
from recommender import BaseRecommenderModel
from utils.read import find_checkpoint
from utils.write import save_obj, store_recommendation

np.random.seed(0)
tf.random.set_seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class MULTIDAE(BaseRecommenderModel):

    def __init__(self, config, params, *args, **kwargs):
        """


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

        self._num_iters = self.params.epochs
        self._factors = self.params.embed_k
        self.l_w = self.params.l_w
        self.l_b = self.params.l_b

        self._restore_epochs = 0

        self._ratings = self._data.train_dataframe_dict

        self._sampler = cs.Sampler(self._ratings, self._random, self._sample_negative_items_empirically)

        self._iteration = 0

        self.evaluator = Evaluator(self._data)
        self._results = []

        ######################################

        self.p_dims = self.params.p_dims
        if self.params.q_dims is None:
            self.q_dims = self.p_dims[::-1]

        self.dims = self.q_dims + self.p_dims[1:]

        self.lam = self.params.reg_lambda
        self._learning_rate = self.params.lr
        self._random = np.random

        self.input_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.dims[0]])
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)

        self.optimizer = tf.train.AdamOptimizer(self.lr)
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
        #TODO
        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights) - 1:
                h = tf.nn.tanh(h)
        return tf.train.Saver(), h

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

    @tf.function
    def train_step(self, batch):
        """
        Apply a single training step on one batch.

        Args:
            batch: batch used for the current train step

        Returns:
            loss value at the current batch
        """
        user, item = batch
        with tf.GradientTape() as tape:

            # Clean Inference
            saver, logits = self.call(inputs=(user, item), training=True)
            log_softmax_var = tf.nn.log_softmax(logits)

            # per-user average negative log-likelihood
            neg_ll = -tf.reduce_mean(tf.reduce_sum(
                log_softmax_var * self.input_ph, axis=1))
            # apply regularization to weights
            regularizer = tf.keras.regularizers.l2(self.lam)
            reg_loss = regularizer(self.weights)
            # tensorflow l2 regularization multiply 0.5 to the l2 norm
            # multiply 2 so that it is back in the same scale
            loss = neg_ll + 2 * reg_loss

            # Loss to be optimized
            loss += reg_loss

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss

    def train(self):

        for it in range(self._restore_epochs, self._num_iters + 1):
            batches = self._sampler.step(self._data.transactions, self.params.batch_size)

            loss = self.one_epoch(batches)

            recs = self.get_recommendations(self._config.top_k)
            self._results.append(self.evaluator.eval(recs))

            print('Epoch {0}/{1} loss {2:.3f}'.format(it, self._num_iters, loss))

            if not (it + 1) % 10:
                store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}_{it + 1}.tsv")