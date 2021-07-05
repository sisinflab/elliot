"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class FISM_model(keras.Model):

    def __init__(self,
                 data,
                 factors,
                 lr,
                 alpha,
                 beta,
                 lambda_,
                 gamma,
                 num_users,
                 num_items,
                 random_seed,
                 name="FISM",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self._data = data
        self._factors = factors
        self._lr = lr
        self._beta = beta
        self._lambda = lambda_
        self._gamma = gamma
        self._alpha = alpha
        self._num_users = num_users
        self._num_items = num_items

        self._history_item_matrix, self._history_lens, self._mask_history_matrix = self.create_history_item_matrix()

        self.initializer = tf.keras.initializers.RandomUniform(-0.001, 0.001)

        self.Bi = tf.Variable(tf.zeros(self._num_items), name='Bi', dtype=tf.float32)
        self.Bu = tf.Variable(tf.zeros(self._num_users), name='Bu', dtype=tf.float32)

        self.Gi = tf.Variable(self.initializer(shape=[self._num_items, self._factors]), name='Gi', dtype=tf.float32)
        self.Gj = tf.Variable(self.initializer(shape=[self._num_items, self._factors]), name='Gj', dtype=tf.float32)

        self.optimizer = tf.optimizers.Adam(self._lr)

        self.loss = keras.losses.BinaryCrossentropy()

    @tf.function
    def call(self, inputs, training=None):
        user, item = inputs
        user_inter = tf.nn.embedding_lookup(self._history_item_matrix, user)
        item_num = tf.nn.embedding_lookup(self._history_lens, user)
        batch_mask_mat = tf.nn.embedding_lookup(self._mask_history_matrix, user)

        user_history = tf.squeeze(tf.nn.embedding_lookup(self.Gi, user_inter))  # batch_size x max_len x embedding_size
        target = tf.squeeze(tf.nn.embedding_lookup(self.Gj, item))  # batch_size x embedding_size
        user_bias = tf.squeeze(tf.nn.embedding_lookup(self.Bu, user))  # batch_size x 1
        item_bias = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))

        similarity = tf.squeeze(tf.matmul(user_history, tf.expand_dims(target, axis=2)))
        similarity = tf.convert_to_tensor(batch_mask_mat, dtype=tf.float32) * similarity
        coeff = tf.pow(item_num, -tf.convert_to_tensor(self._alpha))
        scores = 1 / (1 + tf.math.exp(-(coeff * tf.reduce_sum(similarity, axis=1) + user_bias + item_bias)))
        return scores, user_bias, item_bias, user_history, target

    @tf.function
    def train_step(self, batch):
        user, pos, label = batch
        with tf.GradientTape() as tape:
            # Clean Inference
            output, user_bias, item_bias, source, target = self(inputs=(user, pos), training=True)
            reg_loss = self._lambda * tf.nn.l2_loss(user_bias) + self._gamma * tf.nn.l2_loss(item_bias) + self._beta * tf.reduce_sum(
                [tf.nn.l2_loss(source), tf.nn.l2_loss(target)])
            reg_loss = tf.cast(reg_loss, tf.float64)
            loss = self.loss(output, label) + reg_loss

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss

    @tf.function
    def predict(self, user, **kwargs):
        # user_inters = self._history_item_matrix[user]
        user_inters = tf.nn.embedding_lookup(self._history_item_matrix, user)
        # item_num = self._history_lens[user]
        item_num = tf.nn.embedding_lookup(self._history_lens, user)
        # user_input = user_inters[:item_num]
        user_input = user_inters[:]
        repeats = self._num_items
        user_bias = tf.squeeze(tf.nn.embedding_lookup(self.Bu, user))

        item_num = tf.repeat(item_num, repeats)

        user_history = tf.squeeze(tf.nn.embedding_lookup(self.Gi, user_input))  # inter_num x embedding_size
        user_history = tf.ones([repeats, 1, 1]) * user_history  # target_items x inter_num x embedding_size

        targets = self.Gj  # target_items x embedding_size
        item_bias = self.Bi

        similarity = tf.squeeze(tf.matmul(user_history, tf.expand_dims(targets, axis=2)))
        coeff = tf.pow(tf.cast(item_num, tf.float32), -self._alpha)
        scores = 1 / (1 + tf.math.exp(-(coeff * tf.reduce_sum(similarity, axis=1) + user_bias + item_bias)))

        return scores

    @tf.function
    def batch_predict(self, user_start, user_stop, **kwargs):
        user_inters = tf.nn.embedding_lookup(self._history_item_matrix, range(user_start, user_stop))
        item_num = tf.nn.embedding_lookup(self._history_lens, range(user_start, user_stop))
        user_input = user_inters[:]
        repeats = self._num_items
        user_bias = tf.repeat(tf.nn.embedding_lookup(self.Bu, range(user_start, user_stop)), repeats)

        item_num = tf.repeat(item_num, repeats)

        user_history = tf.squeeze(tf.nn.embedding_lookup(self.Gi, user_input))  # bs x inter_num x embedding_size
        user_history = tf.ones([repeats, 1, 1, 1]) * user_history  # bs x target_items x inter_num x embedding_size
        user_history = tf.reshape(user_history, [user_history.shape[1], user_history.shape[0], user_history.shape[2],
                                                 user_history.shape[3]])
        targets = self.Gj  # target_items x embedding_size
        item_bias = self.Bi

        similarity = tf.squeeze(tf.matmul(user_history, tf.expand_dims(targets, axis=2)))

        batch_eval = similarity.shape[0]
        # coeff = tf.pow(tf.cast(item_num, tf.float32), -self._alpha)
        # scores = 1 / (1 + tf.math.exp(-(coeff * tf.reduce_sum(similarity, axis=1) + user_bias + item_bias)))
        coeff = tf.reshape(tf.pow(tf.cast(item_num, tf.float32), -self._alpha), [batch_eval, self._num_items])
        prod = coeff * tf.reduce_sum(similarity, axis=2)

        return 1 / (1 + tf.math.exp(-(prod + tf.reshape(user_bias, prod.shape) + item_bias)))

    def create_history_item_matrix(self):

        user_ids, item_ids = self._data.sp_i_train.nonzero()[0], self._data.sp_i_train.nonzero()[1]

        row_num, max_col_num = self._num_users, self._num_items
        row_ids, col_ids = user_ids, item_ids

        history_len = np.zeros(row_num, dtype=np.int64)
        for row_id in row_ids:
            history_len[row_id] += 1

        col_num = np.max(history_len)
        history_matrix = np.zeros((row_num, col_num), dtype=np.int64)
        mask_history_matrix = np.zeros((row_num, col_num), dtype=np.int64)
        history_len[:] = 0
        for row_id, col_id in zip(row_ids, col_ids):
            history_matrix[row_id, history_len[row_id]] = col_id
            mask_history_matrix[row_id, history_len[row_id]] = 1
            history_len[row_id] += 1

        # return history_matrix, history_len, mask_history_matrix
        return tf.Variable(history_matrix), tf.Variable(history_len, dtype=tf.float32), tf.Variable(mask_history_matrix,
                                                                                                    dtype=tf.float32)

    @tf.function
    def get_top_k(self, predictions, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, predictions, -np.inf), k=k, sorted=True)

    @tf.function
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


class LatentFactor(tf.keras.layers.Embedding):

    def __init__(self, num_instances, dim, zero_init=False, name=None):

        if zero_init:
            initializer = 'zeros'
        else:
            initializer = 'uniform'
        super(LatentFactor, self).__init__(input_dim=num_instances,
                                           output_dim=dim,
                                           embeddings_initializer=initializer,
                                           name=name)

    def censor(self, censor_id):

        unique_censor_id, _ = tf.unique(censor_id)
        embedding_gather = tf.gather(self.variables[0], indices=unique_censor_id)
        norm = tf.norm(embedding_gather, axis=1, keepdims=True)
        return self.variables[0].scatter_nd_update(indices=tf.expand_dims(unique_censor_id, 1),
                                                   updates=embedding_gather / tf.math.maximum(norm, 0.1))
