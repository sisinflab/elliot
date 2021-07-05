"""
Module description:

Mnih, Andriy, and Russ R. Salakhutdinov. "Probabilistic matrix factorization." Advances in neural information processing systems 20 (2007)

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ProbabilisticMatrixFactorizationModel(keras.Model):
    def __init__(self,
                 num_users,
                 num_items,
                 embed_mf_size,
                 lambda_weights,
                 gaussian_variance,
                 learning_rate=0.01,
                 random_seed=42,
                 name="MF",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)
        self.num_users = num_users
        self.num_items = num_items
        self.embed_mf_size = embed_mf_size
        self.lambda_weights = lambda_weights

        self.initializer = tf.initializers.RandomNormal(stddev=0.01)

        self.user_mf_embedding = keras.layers.Embedding(input_dim=self.num_users, output_dim=self.embed_mf_size,
                                                        embeddings_initializer=self.initializer, name='U_MF',
                                                        embeddings_regularizer=keras.regularizers.l2(self.lambda_weights),
                                                        dtype=tf.float32)
        self.item_mf_embedding = keras.layers.Embedding(input_dim=self.num_items, output_dim=self.embed_mf_size,
                                                        embeddings_regularizer=keras.regularizers.l2(self.lambda_weights),
                                                        embeddings_initializer=self.initializer, name='I_MF',
                                                        dtype=tf.float32)

        self.user_mf_embedding(0)
        self.item_mf_embedding(0)

        self.predict_layer = self.dot_prod
        self.noise = keras.layers.GaussianNoise(gaussian_variance, input_dim=1)

        self.activate = keras.activations.sigmoid
        self.loss = keras.losses.MeanSquaredError()

        self.optimizer = tf.optimizers.Adam(learning_rate)

    @tf.function
    def dot_prod(self, layer_0, layer_1):
        return tf.reduce_sum(layer_0 * layer_1, axis=-1)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        user, item = inputs
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)

        mf_output = self.predict_layer(user_mf_e, item_mf_e)  # [batch_size, embedding_size]

        output = self.activate(mf_output)

        return output

    @tf.function
    def train_step(self, batch):
        user, pos, label = batch
        with tf.GradientTape() as tape:
            # Clean Inference
            output = self.noise(self(inputs=(user, pos), training=True))
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
        user, item = inputs
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)

        mf_output = self.predict_layer(user_mf_e, item_mf_e)  # [batch_size, embedding_size]

        output = self.activate(mf_output)

        return tf.squeeze(output)

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
