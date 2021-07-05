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


class ConvolutionalComponent(tf.keras.Model):
    def __init__(self, channels, kernels, strides, name="ConvolutionalComponent", **kwargs):
        super().__init__(name=name, **kwargs)
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.num_of_nets = len(self.channels) - 1

        if len(self.kernels) != len(self.strides) or self.num_of_nets != (len(self.kernels)):
            raise RuntimeError('channels, kernels and strides don\'t match\n')

        self.initializer = tf.initializers.GlorotUniform()

        self.cnn_network = tf.keras.Sequential()

        # self.cnn_network.add(tf.keras.layers.Input(self.channels[0]))
        for i in range(self.num_of_nets):
            self.cnn_network.add(
                keras.layers.Conv2D(kernel_size=self.kernels[i], filters=self.channels[i + 1], strides=self.strides[i],
                                    use_bias=True,
                                    activation='relu',
                                    kernel_initializer=self.initializer))

    @tf.function
    def call(self, inputs, **kwargs):
        return self.cnn_network(inputs)


class MLPComponent(tf.keras.Model):
    def __init__(self, mlp_layers, dropout=0.2, bn=False, name="MLPComponent", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        self.use_bn = bn

        self.initializer = tf.initializers.GlorotUniform()

        self.mlp_newtork = tf.keras.Sequential()

        for units in self.mlp_layers:
            self.mlp_newtork.add(keras.layers.Dropout(self.dropout))
            self.mlp_newtork.add(keras.layers.Dense(units, activation='linear', kernel_initializer=self.initializer))
            if self.use_bn:
                self.mlp_newtork.add(keras.layers.BatchNormalization())

    @tf.function
    def call(self, inputs, training=False, **kwargs):
        return self.mlp_newtork(inputs, training)


class ConvNeuralMatrixFactorizationModel(keras.Model):
    def __init__(self,
                 num_users, num_items, embedding_size,
                 lr, cnn_channels, cnn_kernels,
                 cnn_strides, dropout_prob, l_w, l_b,
                 random_seed=42,
                 name="ConvNeuralMatrixFactorizationModel",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        tf.random.set_seed(random_seed)
        self.num_users = num_users
        self.num_items = num_items

        self.embedding_size = embedding_size
        self.lr = lr
        self.cnn_channels = cnn_channels
        self.cnn_kernels = cnn_kernels
        self.cnn_strides = cnn_strides
        self.dropout_prob = dropout_prob
        self.l_w = l_w
        self.l_b = l_b

        self.initializer = tf.initializers.GlorotUniform()

        self.user_mf_embedding = keras.layers.Embedding(input_dim=self.num_users, output_dim=self.embedding_size,
                                                        embeddings_initializer=self.initializer, name='U_MF',
                                                        dtype=tf.float32)
        self.item_mf_embedding = keras.layers.Embedding(input_dim=self.num_items, output_dim=self.embedding_size,
                                                        embeddings_initializer=self.initializer, name='I_MF',
                                                        dtype=tf.float32)

        self.conv_layers = ConvolutionalComponent(channels=self.cnn_channels, kernels=self.cnn_kernels,
                                                  strides=self.cnn_strides)

        self.mlp_layers = MLPComponent(mlp_layers=[self.cnn_channels[-1], 1], dropout=self.dropout_prob)

        self.optimizer = tf.optimizers.Adam(self.lr)

    @tf.function
    def call(self, inputs, training=False, **kwargs):
        user, item = inputs
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)

        interaction_map = tf.matmul(user_mf_e, item_mf_e, transpose_a=True,
                                    transpose_b=False)  # batch_size x embedding_size x embedding_soze
        interaction_map = tf.expand_dims(interaction_map, axis=3)
        conv_layers_e = self.conv_layers(interaction_map)
        conv_layers_e = tf.reduce_sum(conv_layers_e, axis=(1, 2))

        prediction = self.mlp_layers(conv_layers_e, training=training)

        return prediction, user_mf_e, item_mf_e

    @tf.function
    def train_step(self, batch):
        user, pos, neg = batch
        with tf.GradientTape() as tape:
            # Clean Inference
            xu_pos, gamma_u, gamma_pos = self(inputs=(user, pos), training=True)
            xu_neg, gamma_u, gamma_neg = self(inputs=(user, neg), training=True)

            difference = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-difference))
            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg)])

            # Loss to be optimized
            loss += reg_loss

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss

    @tf.function
    def predict(self, inputs, training=False, **kwargs):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        output = self.call(inputs=inputs, training=training)
        return output

    @tf.function
    def get_recs(self, inputs, training=False, **kwargs):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        user, item = inputs
        init_shape = user.shape
        user = tf.expand_dims(tf.repeat(user, 1), axis=1)
        item = tf.expand_dims(tf.repeat(item, 1), axis=1)
        # user = tf.reshape(user, shape=(user.shape[1], user.shape[0]))
        # item = tf.reshape(user, shape=(item.shape[1], item.shape[0]))

        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)

        interaction_map = tf.matmul(user_mf_e, item_mf_e, transpose_a=True,
                                    transpose_b=False)  # batch_size x embedding_size x embedding_size
        interaction_map = tf.expand_dims(interaction_map, axis=3)
        conv_layers_e = self.conv_layers(interaction_map)
        conv_layers_e = tf.reduce_sum(conv_layers_e, axis=(1, 2))

        prediction = self.mlp_layers(conv_layers_e, training=training)
        return tf.reshape(prediction, shape=init_shape)

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
