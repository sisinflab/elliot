"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC

import tensorflow as tf
import numpy as np


class HRDRModel(tf.keras.Model, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 user_projection_rating,
                 item_projection_rating,
                 user_review_cnn,
                 item_review_cnn,
                 user_review_attention,
                 item_review_attention,
                 user_final_representation,
                 item_final_representation,
                 dropout,
                 random_seed,
                 name="HRDR",
                 **kwargs
                 ):
        super().__init__()
        tf.random.set_seed(random_seed)

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.user_projection_rating = user_projection_rating
        self.item_projection_rating = item_projection_rating
        self.user_review_cnn = user_review_cnn
        self.item_review_cnn = item_review_cnn
        self.user_review_attention = user_review_attention
        self.item_review_attention = item_review_attention
        self.user_final_representation = user_final_representation
        self.item_final_representation = item_final_representation
        self.dropout = dropout

        self.initializer = tf.initializers.GlorotUniform()

        self.Gu = tf.Variable(self.initializer(shape=[self._num_users, self._factors]), name='Gu', dtype=tf.float32)
        self.Gi = tf.Variable(self.initializer(shape=[self._num_items, self._factors]), name='Gi', dtype=tf.float32)
        self.Bu = tf.Variable(tf.zeros(self._num_users), name='Bu', dtype=tf.float32)
        self.Bi = tf.Variable(tf.zeros(self._num_items), name='Bi', dtype=tf.float32)
        self.Mu = tf.Variable(tf.zeros(1), name='Mu', dtype=tf.float32)

        # mlp for user and item ratings
        self.user_projection_rating_network = tf.keras.Sequential()
        for layer in range(len(self.user_projection_rating) - 1):
            self.user_projection_rating_network.add(tf.keras.layers.Dense(
                units=self.user_projection_rating[layer],
                activation='relu'
            ))
            self.user_projection_rating_network.add(tf.keras.layers.Dropout(self.dropout))
        self.user_projection_rating_network.add(tf.keras.layers.Dense(
            units=self.user_projection_rating[-1],
            activation='relu'
        ))
        self.item_projection_rating_network = tf.keras.Sequential()
        for layer in range(len(self.item_projection_rating) - 1):
            self.item_projection_rating_network.add(tf.keras.layers.Dense(
                units=self.item_projection_rating[layer],
                activation='relu'
            ))
            self.item_projection_rating_network.add(tf.keras.layers.Dropout(self.dropout))
        self.item_projection_rating_network.add(tf.keras.layers.Dense(
            units=self.item_projection_rating[-1],
            activation='relu'
        ))

        # cnn for user and item reviews
        self.user_review_cnn_network = tf.keras.Sequential()
        for layer in range(len(self.user_review_cnn)):
            self.user_review_cnn_network.add(tf.keras.layers.Conv2D(
                filters=self.user_review_cnn[layer],
                kernel_size=[3, 3],
                activation='relu'
            ))
        self.user_review_cnn_network.add(tf.keras.layers.GlobalMaxPool2D())
        self.item_review_cnn_network = tf.keras.Sequential()
        for layer in range(len(self.item_review_cnn)):
            self.item_review_cnn_network.add(tf.keras.layers.Conv2D(
                filters=self.item_review_cnn[layer + 1],
                kernel_size=[3, 3],
                activation='relu'
            ))
        self.item_review_cnn_network.append(tf.keras.layers.GlobalMaxPool2D())

        # attention network for user and item reviews
        self.user_review_attention_network = tf.keras.Sequential()
        for layer in range(len(self.user_review_attention) - 1):
            self.user_review_attention_network.add(tf.keras.layers.Dense(units=self.user_review_attention[layer]))
            self.user_review_attention_network.add(tf.keras.layers.Dropout(self.dropout))
        self.user_review_attention_network.add(tf.keras.layers.Dense(units=self.user_review_attention[-1]))
        self.user_review_attention_network.add(tf.keras.layers.Dense(units=self.user_review_cnn[-1]))
        self.item_review_attention_network = tf.keras.Sequential()
        for layer in range(len(self.item_review_attention) - 1):
            self.item_review_attention_network.add(tf.keras.layers.Dense(units=self.item_review_attention[layer]))
            self.item_review_attention_network.add(tf.keras.layers.Dropout(self.dropout))
        self.item_review_attention_network.add(tf.keras.layers.Dense(units=self.item_review_attention[-1]))
        self.item_review_attention_network.add(tf.keras.layers.Dense(units=self.item_review_cnn[-1]))

        # review-based final representation for users and items
        self.user_final_representation_network = tf.keras.Sequential()
        for layer in range(len(self.user_final_representation) - 1):
            self.user_final_representation_network.add(
                tf.keras.layers.Dense(units=self.user_final_representation[layer]))
            self.user_final_representation_network.add(tf.keras.layers.Dropout(self.dropout))
        self.user_final_representation_network.add(tf.keras.layers.Dense(units=self.user_final_representation[-1]))
        self.item_final_representation_network = tf.keras.Sequential()
        for layer in range(len(self.item_final_representation) - 1):
            self.item_final_representation_network.add(
                tf.keras.layers.Dense(units=self.item_final_representation[layer]))
            self.item_final_representation_network.add(tf.keras.layers.Dropout(self.dropout))
        self.item_final_representation_network.add(tf.keras.layers.Dense(units=self.item_final_representation[-1]))

        # user and item projection matrix for final prediction
        self.W1 = tf.Variable(self.initializer(shape=[1,
                                                      self.user_projection_rating[-1] + self.embed_k +
                                                      self.user_final_representation_network_list[-1]]), name='W1',
                              dtype=tf.float32)

        self.sigmoid = tf.keras.layers.Activation(tf.nn.sigmoid)

        self.optimizer = tf.optimizers.Adam(self._learning_rate)

    @tf.function
    def call(self, inputs, training=None):
        user, item, user_ratings, item_ratings, user_reviews, item_reviews, = inputs
        xu = self.user_projection_rating_network(user_ratings)
        xi = self.item_projection_rating_network(item_ratings)
        ou = self.user_review_cnn_network(user_reviews)
        oi = self.item_review_cnn_network(item_reviews)
        qru = self.user_review_attention_network(xu)
        qri = self.item_review_attention_network(xi)
        au = tf.multiply(ou, qru)
        ai = tf.multiply(oi, qri)

        rui = 0.0

        return rui

    @tf.function
    def predict(self, start, stop):
        return self.Bi + tf.matmul(self.Gu[start:stop], self.Gi, transpose_b=True)

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as t:
            xui, gamma_u, gamma_i, beta_u, beta_i = \
                self.call(inputs=batch, training=True)

            loss = tf.reduce_sum(tf.square(xui - 1.0))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_i),
                                                 tf.nn.l2_loss(beta_u),
                                                 tf.nn.l2_loss(beta_i)],
                                                *[tf.nn.l2_loss(layer) for layer in
                                                  self.user_projection_rating_network.trainable_variables]
                                                * [tf.nn.l2_loss(layer) for layer in
                                                    self.item_projection_rating_network.trainable_variables]
                                                * [tf.nn.l2_loss(layer) for layer in
                                                    self.user_review_cnn_network.trainable_variables]
                                                * [tf.nn.l2_loss(layer) for layer in
                                                    self.item_review_cnn_network.trainable_variables]
                                                * [tf.nn.l2_loss(layer) for layer in
                                                    self.user_review_attention_network.trainable_variables]
                                                * [tf.nn.l2_loss(layer) for layer in
                                                    self.item_review_attention_network.trainable_variables]
                                                * [tf.nn.l2_loss(layer) for layer in
                                                    self.user_final_representation_network.trainable_variables]
                                                * [tf.nn.l2_loss(layer) for layer in
                                                    self.item_final_representation_network.trainable_variables])

            # Loss to be optimized
            loss += reg_loss

        grads = t.gradient(loss, [self.Gu,
                                  self.Gi,
                                  self.Bu,
                                  self.Bi,
                                  self.Mu,
                                  self.W1,
                                  *self.user_projection_rating_network.trainable_variables,
                                  *self.item_projection_rating_network.trainable_variables,
                                  *self.user_review_cnn_network.trainable_variables,
                                  *self.item_review_cnn_network.trainable_variables,
                                  *self.user_review_attention_network.trainable_variables,
                                  *self.item_review_attention_network.trainable_variables,
                                  *self.user_final_representation_network.trainable_variables,
                                  *self.item_final_representation_network.trainable_variables])
        self.optimizer.apply_gradients(zip(grads, [self.Gu,
                                                   self.Gi,
                                                   self.Bu,
                                                   self.Bi,
                                                   self.Mu,
                                                   self.W1,
                                                   *self.user_projection_rating_network.trainable_variables,
                                                   *self.item_projection_rating_network.trainable_variables,
                                                   *self.user_review_cnn_network.trainable_variables,
                                                   *self.item_review_cnn_network.trainable_variables,
                                                   *self.user_review_attention_network.trainable_variables,
                                                   *self.item_review_attention_network.trainable_variables,
                                                   *self.user_final_representation_network.trainable_variables,
                                                   *self.item_final_representation_network.trainable_variables]))

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
