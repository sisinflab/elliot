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


class DeepMatrixFactorizationModel(keras.Model):
    def __init__(self,
                 num_users,
                 num_items,
                 user_mlp,
                 item_mlp,
                 reg,
                 similarity,
                 max_ratings,
                 sp_i_train_ratings,
                 learning_rate=0.01,
                 random_seed=42,
                 name="DMF",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self.num_users = num_users
        self.num_items = num_items
        self.user_mlp = user_mlp
        self.item_mlp = item_mlp
        self.reg = reg
        self.similarity = similarity
        self.max_ratings = max_ratings
        self._sp_i_train_ratings = sp_i_train_ratings

        self.initializer = tf.initializers.RandomNormal(stddev=0.01)

        self.user_embedding = keras.layers.Embedding(input_dim=self.num_users, output_dim=self.num_items,weights=[sp_i_train_ratings.toarray()],
                                                        trainable=False, dtype=tf.float32)
        self.item_embedding = keras.layers.Embedding(input_dim=self.num_items, output_dim=self.num_users,
                                                     weights=[sp_i_train_ratings.T.toarray()],
                                                     trainable=False, dtype=tf.float32)
        self.user_embedding(0)
        self.item_embedding(0)

        self.user_mlp_layers = keras.Sequential()
        for units in user_mlp[:-1]:
            self.user_mlp_layers.add(keras.layers.Dense(units, activation='relu', kernel_initializer=self.initializer))
        self.user_mlp_layers.add(keras.layers.Dense(user_mlp[-1], activation='linear', kernel_initializer=self.initializer))

        self.item_mlp_layers = keras.Sequential()
        for units in item_mlp[:-1]:
            self.item_mlp_layers.add(keras.layers.Dense(units, activation='relu', kernel_initializer=self.initializer))
        self.item_mlp_layers.add(keras.layers.Dense(item_mlp[-1], activation='linear', kernel_initializer=self.initializer))

        if self.similarity == "cosine":
            self.predict_layer = self.cosine
        elif self.similarity == "dot":
            self.predict_layer = self.dot_prod
        else:
            raise NotImplementedError

        self.loss = keras.losses.BinaryCrossentropy()

        self.optimizer = tf.optimizers.Adam(learning_rate)

    @tf.function
    def cosine(self, layer_0, layer_1):
        return tf.reduce_sum(tf.nn.l2_normalize(layer_0, 0) * tf.nn.l2_normalize(layer_1, 0), axis=-1)

    @tf.function
    def dot_prod(self, layer_0, layer_1):
        return tf.reduce_sum(layer_0 * layer_1, axis=-1)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        user, item = inputs
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        user_mlp_output = self.user_mlp_layers(user_e)
        item_mlp_output = self.item_mlp_layers(item_e)
        output = self.predict_layer(user_mlp_output, item_mlp_output)
        return tf.squeeze(output)

    @tf.function
    def train_step(self, batch):
        user, pos, label = batch
        label /= self.max_ratings
        with tf.GradientTape() as tape:
            # Clean Inference
            output = self(inputs=(user, pos), training=True)
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
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        user_mlp_output = self.user_mlp_layers(user_e)
        item_mlp_output = self.item_mlp_layers(item_e)
        output = self.predict_layer(user_mlp_output, item_mlp_output)
        return output

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
