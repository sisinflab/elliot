"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC

import tensorflow as tf
import numpy as np
import os
import random


class DeepCoNNModel(tf.keras.Model, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 l_w,
                 users_vocabulary_features,
                 items_vocabulary_features,
                 textual_words_feature_shape,
                 user_review_cnn_kernel,
                 user_review_cnn_features,
                 item_review_cnn_kernel,
                 item_review_cnn_features,
                 latent_size,
                 dropout_rate,
                 random_seed,
                 name="DeepCoNN",
                 **kwargs
                 ):
        super().__init__()

        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        self.num_users = num_users
        self.num_items = num_items
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.user_review_cnn_kernel = user_review_cnn_kernel
        self.user_review_cnn_features = user_review_cnn_features
        self.item_review_cnn_kernel = item_review_cnn_kernel
        self.item_review_cnn_features = item_review_cnn_features
        self.latent_size = latent_size
        self.dropout_rate = dropout_rate

        self.Vu = tf.expand_dims(tf.convert_to_tensor(users_vocabulary_features, dtype=tf.float32), -1)
        self.Vi = tf.expand_dims(tf.convert_to_tensor(items_vocabulary_features, dtype=tf.float32), -1)

        self.textual_words_feature_shape = textual_words_feature_shape

        # cnn for user and item reviews
        self.user_review_cnn_network = []
        self.user_review_cnn_network.append(
            (tf.Variable(
                initial_value=tf.random.truncated_normal([self.user_review_cnn_kernel[0],
                                                          self.textual_words_feature_shape,
                                                          1,
                                                          self.user_review_cnn_features[0]], stddev=0.1)),
             tf.Variable(initial_value=tf.constant(0.1, shape=[1, self.user_review_cnn_features[0]])))
        )
        for layer in range(1, len(self.user_review_cnn_features)):
            self.user_review_cnn_network.append((tf.Variable(
                initial_value=tf.random.truncated_normal([self.user_review_cnn_kernel[layer],
                                                          self.textual_words_feature_shape,
                                                          self.user_review_cnn_features[layer - 1],
                                                          self.user_review_cnn_features[layer]], stddev=0.1)),
                                                 tf.Variable(initial_value=tf.constant(0.1,
                                                                                       shape=[1,
                                                                                              self.user_review_cnn_features[
                                                                                                  layer]]))))
        self.item_review_cnn_network = []
        self.item_review_cnn_network.append(
            (tf.Variable(
                initial_value=tf.random.truncated_normal([self.item_review_cnn_kernel[0],
                                                          self.textual_words_feature_shape,
                                                          1,
                                                          self.item_review_cnn_features[0]], stddev=0.1)),
             tf.Variable(initial_value=tf.constant(0.1, shape=[1, self.item_review_cnn_features[0]])))
        )
        for layer in range(1, len(self.item_review_cnn_features)):
            self.item_review_cnn_network.append((tf.Variable(
                initial_value=tf.random.truncated_normal([self.item_review_cnn_kernel[0],
                                                          self.textual_words_feature_shape,
                                                          self.item_review_cnn_features[layer - 1],
                                                          self.item_review_cnn_features[layer]], stddev=0.1)),
                                                 tf.Variable(initial_value=tf.constant(0.1,
                                                                                       shape=[1,
                                                                                              self.item_review_cnn_features[
                                                                                                  layer]]))))

        # fully-connected layer for users and items
        self.user_fully_connected = tf.keras.layers.Dense(units=self.latent_size, use_bias=True)
        self.item_fully_connected = tf.keras.layers.Dense(units=self.latent_size, use_bias=True)

        # parameter for FM
        self.W1 = tf.Variable(tf.random.uniform(minval=-0.1, maxval=0.1, shape=[self.latent_size * 2, 1]))
        self.W2 = tf.Variable(tf.random.uniform(minval=-0.1, maxval=0.1, shape=[self.latent_size * 2, 8]))
        self.B = tf.Variable(tf.constant(0.1))

        self.sigmoid = tf.keras.layers.Activation(tf.nn.sigmoid)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.optimizer = tf.optimizers.Adam(self.learning_rate)

    @tf.function
    def conv_users(self, user_reviews):
        user_reviews_features = tf.nn.embedding_lookup(self.Vu, user_reviews)

        out_users = []
        for layer in range(len(self.user_review_cnn_network)):
            out = tf.nn.max_pool(
                tf.nn.conv2d(input=user_reviews_features, filters=self.user_review_cnn_network[layer][0],
                             strides=[1, 1, 1, 1], padding='VALID') + self.user_review_cnn_network[layer][1],
                ksize=[1, user_reviews_features.shape[1] - self.user_review_cnn_kernel[layer] + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
            )
            out_users.append(out)
        out_users = tf.reshape(tf.concat(out_users, axis=1), [user_reviews.shape[0], -1])
        out_users = self.user_fully_connected(out_users)

        return out_users

    @tf.function
    def conv_items(self, item_reviews):
        item_reviews_features = tf.nn.embedding_lookup(self.Vi, item_reviews)

        out_items = []
        for layer in range(len(self.item_review_cnn_network)):
            out = tf.nn.max_pool(
                tf.nn.conv2d(input=item_reviews_features, filters=self.item_review_cnn_network[layer][0],
                             strides=[1, 1, 1, 1], padding='VALID') + self.item_review_cnn_network[layer][1],
                ksize=[1, item_reviews_features.shape[1] - self.item_review_cnn_kernel[layer] + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
            )
            out_items.append(out)
        out_items = tf.reshape(tf.concat(out_items, axis=1), [item_reviews.shape[0], -1])
        out_items = self.user_fully_connected(out_items)

        return out_items

    @tf.function
    def call(self, inputs, training=True):
        user, item, _, user_reviews, item_reviews = inputs
        user_reviews_features = tf.nn.embedding_lookup(self.Vu, user_reviews)
        item_reviews_features = tf.nn.embedding_lookup(self.Vi, item_reviews)

        out_users = []
        for layer in range(len(self.user_review_cnn_network)):
            out = tf.nn.max_pool(
                tf.nn.conv2d(input=user_reviews_features, filters=self.user_review_cnn_network[layer][0],
                             strides=[1, 1, 1, 1], padding='VALID') + self.user_review_cnn_network[layer][1],
                ksize=[1, user_reviews_features.shape[1] - self.user_review_cnn_kernel[layer] + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
            )
            out_users.append(out)
        out_users = tf.reshape(tf.concat(out_users, axis=1), [user.shape[0], -1])
        out_users = self.dropout(self.user_fully_connected(out_users), training=training)

        out_items = []
        for layer in range(len(self.item_review_cnn_network)):
            out = tf.nn.max_pool(
                tf.nn.conv2d(input=item_reviews_features, filters=self.item_review_cnn_network[layer][0],
                             strides=[1, 1, 1, 1], padding='VALID') + self.item_review_cnn_network[layer][1],
                ksize=[1, item_reviews_features.shape[1] - self.item_review_cnn_kernel[layer] + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
            )
            out_items.append(out)
        out_items = tf.reshape(tf.concat(out_items, axis=1), [user.shape[0], -1])
        out_items = self.dropout(self.item_fully_connected(out_items), training=training)

        out = tf.nn.relu(tf.concat([out_users, out_items], axis=-1))
        one = tf.matmul(out, self.W1)

        out_1 = tf.matmul(out, self.W2)
        out_2 = tf.matmul(tf.square(out), tf.square(self.W2))

        out_inter = self.dropout(tf.constant(0.5) * (tf.square(out_1) - out_2), training=training)
        out_inter = tf.reduce_sum(out_inter, -1, keepdims=True)
        out_final = tf.squeeze(self.sigmoid(self.B + out_inter + one))

        return out_final

    @tf.function
    def predict(self, out_users, out_items):
        out = tf.nn.relu(tf.concat([tf.repeat(out_users, repeats=out_items.shape[0], axis=0),
                                    tf.tile(out_items, multiples=tf.constant([out_users.shape[0], 1], tf.int32))],
                                   axis=-1))

        one = tf.matmul(out, self.W1)

        out_1 = tf.matmul(out, self.W2)
        out_2 = tf.matmul(tf.square(out), tf.square(self.W2))

        out_inter = self.dropout(tf.constant(0.5) * (tf.square(out_1) - out_2), training=False)
        out_inter = tf.reduce_sum(out_inter, -1, keepdims=True)
        rui = tf.squeeze(self.sigmoid(self.B + out_inter + one))

        return tf.reshape(rui, [out_users.shape[0], out_items.shape[0]])

    @tf.function
    def train_step(self, batch):
        _, _, r, _, _ = batch
        with tf.GradientTape() as t:
            xui = self(inputs=batch, training=True)
            loss = tf.nn.l2_loss(tf.subtract(xui, r))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(variable) for variable in self.trainable_variables]) * 2

            # Loss to be optimized
            loss += reg_loss

        grads = t.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
