"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class SVDppModel(keras.Model):
    def __init__(self,
                 num_users,
                 num_items,
                 embed_mf_size,
                 lambda_weights,
                 lambda_bias,
                 learning_rate=0.01,
                 random_seed=42,
                 name="FunkSVD",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)
        self.num_users = num_users
        self.num_items = num_items
        self.embed_mf_size = embed_mf_size
        self.lambda_weights = lambda_weights
        self.lambda_bias = lambda_bias

        self.initializer = tf.initializers.GlorotUniform()

        self.user_mf_embedding = keras.layers.Embedding(input_dim=self.num_users, output_dim=self.embed_mf_size,
                                                        embeddings_initializer=self.initializer, name='U_MF',
                                                        embeddings_regularizer=keras.regularizers.l2(self.lambda_weights),
                                                        dtype=tf.float32)
        self.item_mf_embedding = keras.layers.Embedding(input_dim=self.num_items, output_dim=self.embed_mf_size,
                                                        embeddings_initializer=self.initializer, name='I_MF',
                                                        embeddings_regularizer=keras.regularizers.l2(self.lambda_weights),
                                                        dtype=tf.float32)
        self.item_y_embedding = keras.layers.Embedding(input_dim=self.num_items, output_dim=self.embed_mf_size,
                                                        embeddings_initializer=self.initializer, name='Y_MF',
                                                        embeddings_regularizer=keras.regularizers.l2(self.lambda_weights),
                                                        dtype=tf.float32)
        self.user_bias_embedding = keras.layers.Embedding(input_dim=self.num_users, output_dim=1,
                                                          embeddings_initializer=self.initializer, name='U_BIAS',
                                                          embeddings_regularizer=keras.regularizers.l2(self.lambda_bias),
                                                          dtype=tf.float32)
        self.item_bias_embedding = keras.layers.Embedding(input_dim=self.num_items, output_dim=1,
                                                          embeddings_initializer=self.initializer, name='I_BIAS',
                                                          embeddings_regularizer=keras.regularizers.l2(self.lambda_bias),
                                                          dtype=tf.float32)
        self.bias_ = tf.Variable(0., name='GB')

        self.user_mf_embedding(0)
        self.item_mf_embedding(0)
        self.item_y_embedding(0)
        self.user_bias_embedding(0)
        self.item_bias_embedding(0)

        self.loss = keras.losses.MeanSquaredError()

        self.optimizer = tf.optimizers.Adam(learning_rate)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        user, item, pos = inputs
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_bias_e = tf.squeeze(self.user_bias_embedding(user))
        item_bias_e = tf.squeeze(self.item_bias_embedding(item))

        puyj = tf.map_fn(lambda row: tf.math.reduce_mean(self.item_y_embedding.weights[0][row > 0], axis=0), pos)

        dot_prod = tf.reduce_sum((puyj + user_mf_e) * item_mf_e, axis=-1)
        output = dot_prod + user_bias_e + item_bias_e + self.bias_

        return output

    @tf.function
    def train_step(self, batch):
        user, item, label, pos = batch
        with tf.GradientTape() as tape:
            # Clean Inference
            output = self(inputs=(user, item, pos), training=True)
            loss = self.loss(label, output)

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
        user, item, pos = inputs
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_bias_e = tf.squeeze(self.user_bias_embedding(user))
        item_bias_e = tf.squeeze(self.item_bias_embedding(item))

        puyj = tf.expand_dims(tf.map_fn(lambda row: tf.math.reduce_mean(self.item_y_embedding.weights[0][row > 0], axis=0), pos),  axis=1)
        dot_prod = tf.reduce_sum((puyj + user_mf_e) * item_mf_e, axis=-1)
        output = dot_prod + user_bias_e + item_bias_e + self.bias_

        return output

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
