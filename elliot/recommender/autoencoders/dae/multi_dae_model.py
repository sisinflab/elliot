"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Encoder(layers.Layer):
    """Maps user-item interactions to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=200,
                 intermediate_dim=600,
                 dropout_rate=0,
                 regularization_lambda=0.01,
                 random_seed=42,
                 name="encoder",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self.l2_normalizer = layers.Lambda(lambda x: keras.backend.l2_normalize(x, axis=1))
        self.input_dropout = layers.Dropout(dropout_rate)
        self.dense_proj = layers.Dense(intermediate_dim,
                                       activation="tanh",
                                       kernel_initializer=keras.initializers.GlorotNormal(),
                                       kernel_regularizer=keras.regularizers.l2(regularization_lambda))
        self.dense_mean = layers.Dense(latent_dim,
                                       activation="tanh",
                                       kernel_initializer=keras.initializers.GlorotNormal(),
                                       kernel_regularizer=keras.regularizers.l2(regularization_lambda))

    @tf.function
    def call(self, inputs, training=None):
        i_normalized = self.l2_normalizer(inputs, 1)
        i_drop = self.input_dropout(i_normalized, training=training)
        x = self.dense_proj(i_drop)
        z_mean = self.dense_mean(x)
        return z_mean


class Decoder(layers.Layer):
    """Converts z, the encoded vector, back into a uaser interaction vector."""

    def __init__(self, original_dim, intermediate_dim=600, name="decoder", regularization_lambda=0.01,
                 random_seed=42, **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)
        self.dense_proj = layers.Dense(intermediate_dim,
                                       activation="tanh",
                                       kernel_initializer=keras.initializers.GlorotNormal(),
                                       kernel_regularizer=keras.regularizers.l2(regularization_lambda))
        self.dense_output = layers.Dense(original_dim,
                                         kernel_initializer=keras.initializers.GlorotNormal(),
                                         kernel_regularizer=keras.regularizers.l2(regularization_lambda))

    @tf.function
    def call(self, inputs, **kwargs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class DenoisingAutoEncoder(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 original_dim,
                 intermediate_dim=600,
                 latent_dim=200,
                 learning_rate=0.001,
                 dropout_rate=0,
                 regularization_lambda=0.01,
                 random_seed=42,
                 name="DenoisingAutoEncoder",
                 **kwargs
    ):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                               intermediate_dim=intermediate_dim,
                               dropout_rate=dropout_rate,
                               regularization_lambda=regularization_lambda,
                               random_seed=random_seed)
        self.decoder = Decoder(original_dim,
                               intermediate_dim=intermediate_dim,
                               regularization_lambda=regularization_lambda,
                               random_seed=random_seed)
        self.optimizer = tf.optimizers.Adam(learning_rate)

    def get_config(self):
        raise NotImplementedError

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        z_mean = self.encoder(inputs, training=training)
        reconstructed = self.decoder(z_mean)
        return reconstructed

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:

            # Clean Inference
            logits = self.call(inputs=batch, training=True)
            log_softmax_var = tf.nn.log_softmax(logits)

            # per-user average negative log-likelihood
            loss = -tf.reduce_mean(tf.reduce_sum(
                log_softmax_var * batch, axis=1))

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

        logits = self.call(inputs=inputs, training=training)
        log_softmax_var = tf.nn.log_softmax(logits)
        return log_softmax_var

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)

