"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import numpy as np
import tensorflow as tf
from tensorflow import keras


class ACFModel(keras.Model):
    def __init__(self, factors=200,
                 layers_component=(64, 1),
                 layers_item=(64, 1),
                 learning_rate=0.001,
                 l_w=0,
                 feature_shape=(49, 2048),
                 num_users=100,
                 num_items=100,
                 random_seed=42,
                 name="ACF",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self._factors = factors
        self.l_w = l_w
        self.feature_shape = feature_shape
        self._learning_rate = learning_rate
        self._num_items = num_items
        self._num_users = num_users

        self.layers_component = layers_component
        self.layers_item = layers_item

        self.initializer = tf.initializers.RandomNormal(stddev=0.01)
        self.initializer_attentive = tf.initializers.GlorotUniform()

        self.Gu = tf.Variable(self.initializer(shape=[self._num_users, self._factors]), name='Gu', dtype=tf.float32)
        self.Gi = tf.Variable(self.initializer(shape=[self._num_items, self._factors]), name='Gi', dtype=tf.float32)
        self.Pi = tf.Variable(
            self.initializer(shape=[self._num_items, self._factors]),
            name='Tu', dtype=tf.float32)

        self.component_weights, self.item_weights = self._build_attention_weights()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)

    def _build_attention_weights(self):
        component_dict = dict()
        items_dict = dict()

        for c in range(len(self.layers_component)):
            # the inner layer has all components
            if c == 0:
                component_dict['W_{}_u'.format(c)] = tf.Variable(
                    self.initializer_attentive(shape=[self._factors, self.layers_component[c]]),
                    name='W_{}_u'.format(c),
                    dtype=tf.float32
                )
                component_dict['W_{}_i'.format(c)] = tf.Variable(
                    self.initializer_attentive(shape=[self.feature_shape[-1], self.layers_component[c]]),
                    name='W_{}_i'.format(c),
                    dtype=tf.float32
                )
                component_dict['b_{}'.format(c)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_component[c]]),
                    name='b_{}'.format(c),
                    dtype=tf.float32
                )
            else:
                component_dict['W_{}'.format(c)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_component[c - 1], self.layers_component[c]]),
                    name='W_{}_u'.format(c),
                    dtype=tf.float32
                )
                component_dict['b_{}'.format(c)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_component[c]]),
                    name='b_{}'.format(c),
                    dtype=tf.float32
                )

        for i in range(len(self.layers_item)):
            # the inner layer has all components
            if i == 0:
                items_dict['W_{}_u'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self._factors, self.layers_item[i]]),
                    name='W_{}_u'.format(i),
                    dtype=tf.float32
                )
                items_dict['W_{}_iv'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self._factors, self.layers_item[i]]),
                    name='W_{}_iv'.format(i),
                    dtype=tf.float32
                )
                items_dict['W_{}_ip'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self._factors, self.layers_item[i]]),
                    name='W_{}_ip'.format(i),
                    dtype=tf.float32
                )
                items_dict['W_{}_ix'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self.feature_shape[-1], self.layers_item[i]]),
                    name='W_{}_ix'.format(i),
                    dtype=tf.float32
                )
                items_dict['b_{}'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_item[i]]),
                    name='b_{}'.format(i),
                    dtype=tf.float32
                )
            else:
                items_dict['W_{}'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_item[i - 1], self.layers_item[i]]),
                    name='W_{}_u'.format(i),
                    dtype=tf.float32
                )
                items_dict['b_{}'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_item[i]]),
                    name='b_{}'.format(i),
                    dtype=tf.float32
                )
        return component_dict, items_dict

    @tf.function
    def _calculate_beta_alpha(self, g_u, g_i, p_i, f_i):
        # calculate beta
        b_i_l = tf.expand_dims(
            tf.expand_dims(tf.matmul(tf.expand_dims(g_u, 0), self.component_weights['W_{}_u'.format(0)]), 1), 1) + \
                tf.matmul(f_i, self.component_weights['W_{}_i'.format(0)]) + \
                self.component_weights['b_{}'.format(0)]
        b_i_l = tf.nn.relu(b_i_l)
        for c in range(1, len(self.layers_component)):
            b_i_l = tf.matmul(b_i_l, self.component_weights['W_{}'.format(c)]) + \
                    self.component_weights['b_{}'.format(c)]

        b_i_l = tf.nn.softmax(b_i_l, 2)
        all_x_l = tf.reduce_sum(tf.multiply(b_i_l, f_i), axis=2)

        # calculate alpha
        a_i_l = tf.expand_dims(tf.matmul(tf.expand_dims(g_u, 0), self.item_weights['W_{}_u'.format(0)]), 1) + \
                tf.matmul(tf.expand_dims(g_i, 0), self.item_weights['W_{}_iv'.format(0)]) + \
                tf.matmul(tf.expand_dims(p_i, 0), self.item_weights['W_{}_ip'.format(0)]) + \
                tf.matmul(all_x_l, self.item_weights['W_{}_ix'.format(0)]) + \
                self.item_weights['b_{}'.format(0)]
        a_i_l = tf.nn.relu(a_i_l)
        for c in range(1, len(self.layers_item)):
            a_i_l = tf.matmul(a_i_l, self.item_weights['W_{}'.format(c)]) + \
                    self.item_weights['b_{}'.format(c)]

        a_i_l = tf.nn.softmax(a_i_l, 2)
        all_a_i_l = tf.reduce_sum(tf.multiply(a_i_l, tf.expand_dims(p_i, 0)), 2)
        g_u_p = g_u + all_a_i_l

        return tf.squeeze(g_u_p)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        user, item, user_pos, f_u_i_pos = inputs
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        p_i = tf.squeeze(tf.nn.embedding_lookup(self.Pi, item))

        gamma_i_u_pos = tf.nn.embedding_lookup(self.Gi, user_pos)
        p_i_u_pos = tf.nn.embedding_lookup(self.Pi, user_pos)

        gamma_u_p = self._calculate_beta_alpha(gamma_u, gamma_i_u_pos, p_i_u_pos, f_u_i_pos)
        xui = tf.reduce_sum(gamma_u_p * gamma_i)

        return xui, gamma_u, gamma_i, p_i

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as t:
            user, pos, neg, user_pos, feat_pos = batch
            xu_pos, gamma_u, gamma_pos, p_i_pos = self((user, pos, user_pos, feat_pos), training=True)
            xu_neg, _, gamma_neg, p_i_neg = self((user, neg, user_pos, feat_pos), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg),
                                                 tf.nn.l2_loss(p_i_pos),
                                                 tf.nn.l2_loss(p_i_neg),
                                                 *[tf.nn.l2_loss(value)
                                                   for _, value in self.component_weights.items()],
                                                 *[tf.nn.l2_loss(value)
                                                   for _, value in self.item_weights.items()]])
            # Loss to be optimized
            loss += reg_loss

        params = [self.Gu,
                  self.Gi,
                  self.Pi,
                  *[value for _, value in self.component_weights.items()],
                  *[value for _, value in self.item_weights.items()]]

        grads = t.gradient(loss, params)
        self.optimizer.apply_gradients(zip(grads, params))

        return loss

    @tf.function
    def predict(self, user, user_pos, feat_pos):
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))

        gamma_i_u_pos = tf.expand_dims(tf.nn.embedding_lookup(self.Gi, user_pos), 0)
        p_i_u_pos = tf.expand_dims(tf.nn.embedding_lookup(self.Pi, user_pos), 0)

        gamma_u_p = self._calculate_beta_alpha(gamma_u, gamma_i_u_pos, p_i_u_pos, tf.expand_dims(feat_pos, 0))
        return tf.squeeze(tf.matmul(tf.expand_dims(gamma_u_p, 0), self.Gi, transpose_b=True))

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)

    def get_config(self):
        raise NotImplementedError
