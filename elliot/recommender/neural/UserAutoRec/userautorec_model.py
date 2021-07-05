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


class Encoder(tf.keras.layers.Layer):

    def __init__(self, hidden_neuron=200,
                 regularization=0.01,
                 name="encoder",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense = tf.keras.layers.Dense(hidden_neuron,
                                           activation="sigmoid",
                                           kernel_initializer=keras.initializers.GlorotNormal(),
                                           kernel_regularizer=keras.regularizers.l2(regularization),
                                           bias_initializer=keras.initializers.Ones())

    @tf.function
    def call(self, inputs, training=None):
        x = self.dense(inputs)
        return x


class Decoder(tf.keras.layers.Layer):

    def __init__(self, num_items, name="decoder", regularization=0.01, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_output = tf.keras.layers.Dense(num_items,
                                                  activation='linear',
                                                  kernel_initializer=keras.initializers.GlorotNormal(),
                                                  kernel_regularizer=keras.regularizers.l2(regularization),
                                                  bias_initializer=keras.initializers.Ones())

    @tf.function
    def call(self, inputs, **kwargs):
        x = self.dense_output(inputs)
        return x


class UserAutoRecModel(keras.Model):
    def __init__(self,
                 data,
                 num_users,
                 num_items,
                 lr,
                 hidden_neuron,
                 l_w,
                 random_seed=42,
                 name="UserAutoRec",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self.data = data
        self.num_users = num_users
        self.num_items = num_items
        self.lr = lr
        self.hidden_neuron = hidden_neuron
        self.l_w = l_w

        self.encoder = Encoder(hidden_neuron=self.hidden_neuron,
                               regularization=self.l_w
                               )
        self.decoder = Decoder(num_items=self.num_items,
                               regularization=self.l_w)

        self.optimizer = tf.optimizers.Adam(self.lr)

    def get_config(self):
        raise NotImplementedError

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        encoded = self.encoder(inputs, training=training)
        reconstructed = self.decoder(encoded)
        return reconstructed

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            # Clean Inference
            reconstructed = self.call(inputs=batch, training=True)

            # Observing the contribution of only training ratings
            reconstructed = reconstructed * tf.sign(batch)
            # error
            error = batch - reconstructed
            loss = tf.reduce_mean(tf.reduce_sum(error**2, axis=1))

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

        scores = self.call(inputs=inputs, training=training)
        return scores

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
