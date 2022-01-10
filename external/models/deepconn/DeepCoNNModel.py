"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC

import tensorflow as tf
import numpy as np


class DeepCoNNModel(tf.keras.Model, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 l_w,
                 vocabulary_features,
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
        tf.random.set_seed(random_seed)

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

        self.V = tf.expand_dims(tf.convert_to_tensor(vocabulary_features, dtype=tf.float32), -1)

        self.textual_words_feature_shape = self.V.shape[-1]

        self.initializer = tf.initializers.GlorotUniform()

        # cnn for user and item reviews
        self.user_review_cnn_network = []
        self.user_review_cnn_network.append(
            (tf.Variable(
                initial_value=self.initializer([self.user_review_cnn_kernel[0],
                                                self.textual_words_feature_shape,
                                                1,
                                                self.user_review_cnn_features[0]])),
             tf.Variable(initial_value=tf.zeros([1, self.user_review_cnn_features[0]])))
        )
        for layer in range(1, len(self.user_review_cnn_features)):
            self.user_review_cnn_network.append((tf.Variable(
                initial_value=self.initializer([self.user_review_cnn_kernel[layer],
                                                self.textual_words_feature_shape,
                                                1,
                                                self.user_review_cnn_features[layer]])),
                                                 tf.Variable(initial_value=tf.zeros(
                                                     [1, self.user_review_cnn_features[layer]]))))
        self.item_review_cnn_network = []
        self.item_review_cnn_network.append(
            (tf.Variable(
                initial_value=self.initializer([self.item_review_cnn_kernel[0],
                                                self.textual_words_feature_shape,
                                                1,
                                                self.item_review_cnn_features[0]])),
             tf.Variable(initial_value=tf.zeros([1, self.item_review_cnn_features[0]])))
        )
        for layer in range(1, len(self.item_review_cnn_features)):
            self.item_review_cnn_network.append((tf.Variable(
                initial_value=self.initializer([self.item_review_cnn_kernel[layer],
                                                self.textual_words_feature_shape,
                                                1,
                                                self.item_review_cnn_features[layer]])),
                                                 tf.Variable(initial_value=tf.zeros(
                                                     [1, self.item_review_cnn_features[layer]]))))

        # fully-connected layer for users and items
        self.user_fully_connected = tf.keras.layers.Dense(units=self.latent_size, use_bias=True)
        self.item_fully_connected = tf.keras.layers.Dense(units=self.latent_size, use_bias=True)

        # parameter for FM
        self.W = tf.Variable(tf.random.normal([self.latent_size * 2, 8]))
        self.fm_fully_connected = tf.keras.layers.Dense(1)
        self.B = tf.Variable(tf.zeros(1))

        self.sigmoid = tf.keras.layers.Activation(tf.nn.sigmoid)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.optimizer = tf.optimizers.RMSprop(self.learning_rate)

    @tf.function
    def conv_users(self, user_reviews):
        user_reviews_features = tf.nn.embedding_lookup(self.V, user_reviews)

        out_users = []
        for layer in range(len(self.user_review_cnn_network)):
            out = tf.reduce_max(
                tf.nn.conv2d(input=user_reviews_features, filters=self.user_review_cnn_network[layer][0],
                             strides=[1, 1, 1, 1], padding='VALID') + self.user_review_cnn_network[layer][1], axis=1)
            out_users.append(out)
        out_users = tf.reshape(tf.concat(out_users, axis=1), [user_reviews.shape[0], -1])
        out_users = self.user_fully_connected(out_users)

        return out_users

    @tf.function
    def conv_items(self, item_reviews):
        item_reviews_features = tf.nn.embedding_lookup(self.V, item_reviews)

        out_items = []
        for layer in range(len(self.item_review_cnn_network)):
            out = tf.reduce_max(
                tf.nn.conv2d(input=item_reviews_features, filters=self.item_review_cnn_network[layer][0],
                             strides=[1, 1, 1, 1], padding='VALID') + self.item_review_cnn_network[layer][1], axis=1)
            out_items.append(out)
        out_items = tf.reshape(tf.concat(out_items, axis=1), [item_reviews.shape[0], -1])
        out_items = self.user_fully_connected(out_items)

        return out_items

    @tf.function
    def call(self, inputs, training=True):
        user, item, _, user_reviews, item_reviews = inputs
        user_reviews_features = tf.nn.embedding_lookup(self.V, user_reviews)
        item_reviews_features = tf.nn.embedding_lookup(self.V, item_reviews)

        out_users = []
        for layer in range(len(self.user_review_cnn_network)):
            out = tf.reduce_max(
                tf.nn.conv2d(input=user_reviews_features, filters=self.user_review_cnn_network[layer][0],
                             strides=[1, 1, 1, 1], padding='VALID') + self.user_review_cnn_network[layer][1], axis=1)
            out_users.append(out)
        out_users = tf.reshape(tf.concat(out_users, axis=1), [user.shape[0], -1])
        out_users = self.dropout(self.user_fully_connected(out_users), training=training)

        out_items = []
        for layer in range(len(self.item_review_cnn_network)):
            out = tf.reduce_max(
                tf.nn.conv2d(input=item_reviews_features, filters=self.item_review_cnn_network[layer][0],
                             strides=[1, 1, 1, 1], padding='VALID') + self.item_review_cnn_network[layer][1], axis=1)
            out_items.append(out)
        out_items = tf.reshape(tf.concat(out_items, axis=1), [user.shape[0], -1])
        out_items = self.dropout(self.item_fully_connected(out_items), training=training)

        out = tf.concat([out_users, out_items], axis=-1)

        out_1 = tf.reduce_sum(tf.math.pow(tf.matmul(out, self.W), 2), 1, keepdims=True)
        out_2 = tf.reduce_sum(tf.matmul(tf.math.pow(out, 2), tf.math.pow(self.W, 2)), 1, keepdims=True)

        out_inter = tf.constant(0.5) * (out_1 - out_2)
        out_lin = self.fm_fully_connected(out)
        out_final = tf.squeeze(self.sigmoid(self.B + out_inter + out_lin))

        return out_final

    @tf.function
    def predict(self, out_users, out_items):
        out = tf.concat([tf.repeat(out_users, repeats=out_items.shape[0], axis=0),
                         tf.tile(out_items, multiples=tf.constant([out_users.shape[0], 1], tf.int32))], axis=-1)
        out_1 = tf.reduce_sum(tf.math.pow(tf.matmul(out, self.W), 2), 1, keepdims=True)
        out_2 = tf.reduce_sum(tf.matmul(tf.math.pow(out, 2), tf.math.pow(self.W, 2)), 1, keepdims=True)

        out_inter = tf.constant(0.5) * (out_1 - out_2)
        out_lin = self.fm_fully_connected(out)
        rui = tf.squeeze(self.sigmoid(self.B + out_inter + out_lin))
        return tf.reshape(rui, [out_users.shape[0], out_items.shape[0]])

    @tf.function
    def train_step(self, batch):
        _, _, r, _, _ = batch
        with tf.GradientTape() as t:
            xui = self(inputs=batch, training=True)
            loss = tf.reduce_mean(tf.square(xui - r))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(variable) for variable in self.trainable_variables])

            # Loss to be optimized
            loss += reg_loss

        grads = t.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
