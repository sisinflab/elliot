"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    @tf.function
    def call(self, inputs):
        self.z_mean, self.z_log_var = inputs
        self.batch = tf.shape(self.z_mean)[0]
        self.dim = tf.shape(self.z_mean)[1]
        self.epsilon = tf.keras.backend.random_normal(shape=(self.batch, self.dim))
        return self.z_mean + tf.exp(0.5 * self.z_log_var) * self.epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=200,
                 intermediate_dim=600,
                 dropout_rate=0,
                 regularization_lambda=0.01,
                 name="encoder",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.l2_normalizer = layers.Lambda(lambda x: keras.backend.l2_normalize(x, axis=1))
        self.input_dropout = layers.Dropout(dropout_rate)
        self.dense_proj = layers.Dense(intermediate_dim,
                                       activation="tanh",
                                       kernel_initializer=keras.initializers.GlorotNormal(),
                                       kernel_regularizer=keras.regularizers.l2(regularization_lambda))
        self.dense_mean = layers.Dense(latent_dim,
                                       kernel_initializer=keras.initializers.GlorotNormal(),
                                       kernel_regularizer=keras.regularizers.l2(regularization_lambda))
        self.dense_log_var = layers.Dense(latent_dim,
                                       kernel_initializer=keras.initializers.GlorotNormal(),
                                       kernel_regularizer=keras.regularizers.l2(regularization_lambda))
        self.sampling = Sampling()


    @tf.function
    def call(self, inputs, training=None):
        self.i_normalized = self.l2_normalizer(inputs, 1)
        self.i_drop = self.input_dropout(self.i_normalized, training=training)
        self.x = self.dense_proj(self.i_drop)
        self.z_mean = self.dense_mean(self.x)
        self.z_log_var = self.dense_log_var(self.x)
        self.z = self.sampling((self.z_mean, self.z_log_var))
        return self.z_mean, self.z_log_var, self.z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=600, name="decoder", regularization_lambda=0.01, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim,
                                       activation="tanh",
                                       kernel_initializer=keras.initializers.GlorotNormal(),
                                       kernel_regularizer=keras.regularizers.l2(regularization_lambda))
        self.dense_output = layers.Dense(original_dim,
                                       kernel_initializer=keras.initializers.GlorotNormal(),
                                       kernel_regularizer=keras.regularizers.l2(regularization_lambda))

    @tf.function
    def call(self, inputs, **kwargs):
        # x = self.dense_proj(inputs)
        return self.dense_output(self.dense_proj(inputs))


class KnowledgeAwareVariationalAutoEncoder(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 output_dim,
                 item_factors,
                 intermediate_dim=600,
                 latent_dim=200,
                 learning_rate=0.001,
                 dropout_rate=0,
                 regularization_lambda=0.01,
                 alpha=1.0,
                 name="KnowledgeAwareVariationalAutoEncoder",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(42)

        self.alpha = alpha
        self.item_embedding = keras.layers.Embedding(input_dim=item_factors.shape[0], output_dim=item_factors.shape[1],
                                                     weights=[tf.linalg.l2_normalize(item_factors)],
                                                     embeddings_regularizer=keras.regularizers.l2(regularization_lambda),
                                                     trainable=True, dtype=tf.float32)
        # needed for initialization
        self.item_embedding(0)
        self.encoder = Encoder(latent_dim=latent_dim,
                               intermediate_dim=intermediate_dim,
                               dropout_rate=dropout_rate,
                               regularization_lambda=regularization_lambda)
        self.decoder = Decoder(output_dim,
                               intermediate_dim=intermediate_dim,
                               regularization_lambda=regularization_lambda)
        self.optimizer = tf.optimizers.Adam(learning_rate)
        tf.Graph().finalize()

    def get_config(self):
        raise NotImplementedError

    def convert_function(self, row):
        return tf.math.reduce_mean(self.item_embedding.weights[0][row > 0], axis=0)

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        # tensor_list = tf.map_fn(lambda row: tf.math.reduce_mean(self.item_embedding.weights[0][row > 0], axis=0), inputs)
        # a = tf.where(tf.not_equal(inputs, 0))
        # b = self.item_embedding(a[:,1])
        # tensor_list = tf.map_fn(lambda p: tf.math.reduce_mean(b[a[:, 0] == p], axis=0), tf.cast(tf.unique(a[:,0]).y, tf.int64))
        # for p in tf.unique(a[:,0]).y:
        #     tensor_list.append(tf.math.reduce_mean(b[a[:, 0] == p], axis=0))
        # new_input = tf.convert_to_tensor(tensor_list)

        # self.z_mean, self.z_log_var, self.z = self.encoder(tf.map_fn(lambda row: tf.math.reduce_mean(self.item_embedding.weights[0][row > 0], axis=0), inputs), training=training)
        # self.z_mean, self.z_log_var, self.z = self.encoder(tf.concat([tf.linalg.l2_normalize(inputs, axis=-1), self.alpha * tf.linalg.l2_normalize(tf.map_fn(lambda row: tf.math.reduce_mean(self.item_embedding.weights[0][row > 0], axis=0), inputs), axis=-1)], -1), training=training)

        self.z_mean, self.z_log_var, self.z = self.encoder(tf.concat([inputs,
                                                                      self.alpha * tf.linalg.l2_normalize(tf.map_fn(
                                                                          lambda row: tf.math.reduce_mean(
                                                                              self.item_embedding.weights[0][row > 0],
                                                                              axis=0), inputs), axis=-1)], -1),
                                                           training=training)

        self.reconstructed = self.decoder(self.z)
        # # Add KL divergence regularization loss.
        self.kl_loss = -0.5 * tf.reduce_mean(
            self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var) + 1
        )

        # self.add_loss(kl_loss)
        return self.reconstructed, self.kl_loss

    @tf.function
    def train_step(self, batch, anneal_ph=0.0, **kwargs):
        with tf.GradientTape() as tape:

            # Clean Inference
            self.logits, KL = self.call(inputs=batch, training=True)
            self.log_softmax_var = tf.nn.log_softmax(self.logits)

            # per-user average negative log-likelihood
            self.neg_ll = -tf.reduce_mean(tf.reduce_sum(
                self.log_softmax_var * batch, axis=-1))

            loss = self.neg_ll + anneal_ph * KL

        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_weights), self.trainable_weights))

        return loss

    @tf.function
    def predict(self, inputs, training=False, **kwargs):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """

        logits, _ = self.call(inputs=inputs, training=training)
        log_softmax_var = tf.nn.log_softmax(logits)
        return log_softmax_var

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
