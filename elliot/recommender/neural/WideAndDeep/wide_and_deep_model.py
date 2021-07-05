"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class WideAndDeepModel(tf.keras.Model):
    def __init__(self, data, num_users, num_items, embedding_size, mlp_hidden_size, dropout_prob, lr, l_w, l_b,
                 random_seed=42,
                 name="WideAndDeepModel",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self._data = data
        self._num_users = num_users
        self._num_items = num_items
        self._embedding_size = embedding_size
        self._mlp_hidden_size = mlp_hidden_size
        self._dropout_prob = dropout_prob
        self._lr = lr
        self._l_w = l_w
        self._l_b = l_b
        self._all_item_enc = None
        self._all_item_features_enc = None

        # List of possible embeddings
        self._sparse_dimensions = [self._num_users, self._num_items] + [sp_i_feature.shape[1] for sp_i_feature in self._data.sp_i_features]  # the last component should be replaced by a list of possible features

        self._num_type_of_categorical_features = len(self._data.sp_i_features)

        self._size_list = [self._embedding_size * (self._num_type_of_categorical_features + 2)] + list(
            self._mlp_hidden_size)  # +2 because we have user and item id

        self.initializer = tf.initializers.GlorotUniform()
        # Regularizers
        self.regularizer = keras.regularizers.l2(self._l_w)
        self.bias_regularizer = keras.regularizers.l2(self._l_b)

        # Wide
        self._len_sparse_dimension = sum(self._sparse_dimensions)
        self.wide = keras.layers.Dense(1, use_bias=True, kernel_regularizer=self.regularizer,
                                       bias_regularizer=self.bias_regularizer)

        # Deep
        self.deep = keras.Sequential()
        for units in self._size_list[:-1]:
            self.deep.add(
                keras.layers.Dense(units, use_bias=True, activation='relu', kernel_initializer=self.initializer,
                                   kernel_regularizer=self.regularizer, bias_regularizer=self.bias_regularizer))
        self.deep.add(keras.layers.Dense(self._size_list[-1], use_bias=True, activation='linear',
                                         kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                         bias_regularizer=self.bias_regularizer))

        self.predict_layer = keras.layers.Dense(1, use_bias=True, activation='sigmoid',
                                                kernel_regularizer=self.regularizer,
                                                bias_regularizer=self.bias_regularizer)

        self.loss = keras.losses.BinaryCrossentropy()

        self.optimizer = tf.optimizers.Adam(self._lr)

    # @tf.function
    def call(self, inputs, training=False, **kwargs):
        _, _, s = inputs

        # Wide
        wide_part = self.wide(s)

        # Deep
        deep_part = self.deep(s)

        concat = tf.concat([wide_part, deep_part], axis=1)

        predict = self.predict_layer(concat)

        return predict

    # @tf.function
    def train_step(self, batch):
        u, i, s, label = batch
        with tf.GradientTape() as tape:
            # # Clean Inference
            predict = self(inputs=(u, i, s), training=True)

            loss = self.loss(label, predict)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss

    # @tf.function
    def predict(self, user, **kwargs):
        u_enc = self._data.user_encoder.transform([[user]])

        if self._all_item_enc is None:
            self._all_item_enc = tf.convert_to_tensor(self._data.item_encoder.transform((np.reshape(np.arange(self._num_items), newshape=(self._num_items, 1)))).todense())
        if self._all_item_features_enc is None:
            # f_one_hot = list(itertools.chain.from_iterable([sp_i_feature.todense() for sp_i_feature in self._data.sp_i_features]))
            self._all_item_features_enc = tf.convert_to_tensor(self._data.sp_i_features[0].todense()) # Need to be scrolled

        u_enc = tf.repeat(u_enc.toarray(), self._num_items, axis=0)

        s = tf.concat([tf.cast(u_enc, tf.float32), tf.cast(self._all_item_enc, tf.float32),  tf.cast(self._all_item_features_enc, tf.float32)], axis=1)

        return self(inputs=(None, None, s), transpose_b=True)

    def get_user_recs(self, user, k=100):
        user_items = self._data.train_dict[user].keys()

        predictions = {i: self(inputs=(user, i, self.get_sparse(user, i))) for i in self._data.items if i not in user_items}
        indices, values = zip(*predictions.items())
        indices = np.array(indices)
        values = np.array(tf.squeeze(values))
        partially_ordered_preds_indices = np.argpartition(values, -k)[-k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]

    def get_sparse(self, u, i):
        u_one_hot = [0 for _ in range(self._num_users)]
        u_one_hot[self._data.public_users[u]] = 1
        i_one_hot = [0 for _ in range(self._num_items)]
        i_one_hot[self._data.public_items[i]] = 1
        f_one_hot = self._data.sp_i_features.getrow(self._data.public_items[i]).toarray()[0].tolist()
        s = []
        s += u_one_hot
        s += i_one_hot
        s += f_one_hot
        return tf.reshape(tf.convert_to_tensor(np.array(s)), shape=(1, len(s)))

    # @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)