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


class NeuralMatrixFactorizationModel(keras.Model):
    def __init__(self,
                 num_users,
                 num_items,
                 embed_mf_size, embed_mlp_size, mlp_hidden_size, dropout, is_mf_train,
                 is_mlp_train, learning_rate=0.01,
                 random_seed=42,
                 name="NeuralMatrixFactorizationModel",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)
        self.num_users = num_users
        self.num_items = num_items
        self.embed_mf_size = embed_mf_size
        self.embed_mlp_size = embed_mlp_size
        self.mlp_hidden_size = mlp_hidden_size
        self.dropout = dropout
        self.is_mf_train = is_mf_train
        self.is_mlp_train = is_mlp_train

        self.initializer = tf.initializers.GlorotUniform()

        self.user_mf_embedding = keras.layers.Embedding(input_dim=self.num_users, output_dim=self.embed_mf_size,
                                                        embeddings_initializer=self.initializer, name='U_MF',
                                                        dtype=tf.float32)
        self.item_mf_embedding = keras.layers.Embedding(input_dim=self.num_items, output_dim=self.embed_mf_size,
                                                        embeddings_initializer=self.initializer, name='I_MF',
                                                        dtype=tf.float32)
        self.user_mlp_embedding = keras.layers.Embedding(input_dim=self.num_users, output_dim=self.embed_mlp_size,
                                                         embeddings_initializer=self.initializer, name='U_MLP',
                                                         dtype=tf.float32)
        self.item_mlp_embedding = keras.layers.Embedding(input_dim=self.num_items, output_dim=self.embed_mlp_size,
                                                         embeddings_initializer=self.initializer, name='I_MLP',
                                                         dtype=tf.float32)
        self.user_mf_embedding(0)
        self.user_mlp_embedding(0)
        self.item_mf_embedding(0)
        self.item_mlp_embedding(0)

        self.mlp_layers = keras.Sequential()

        for units in mlp_hidden_size:
            self.mlp_layers.add(keras.layers.Dropout(dropout))
            self.mlp_layers.add(keras.layers.Dense(units, activation='relu'))

        if self.is_mf_train and self.is_mlp_train:
            self.predict_layer = keras.layers.Dense(1, input_dim=self.embed_mf_size + self.mlp_hidden_size[-1])
        elif self.is_mf_train:
            self.predict_layer = keras.layers.Dense(1, input_dim=self.embed_mf_size)
        elif self.is_mlp_train:
            self.predict_layer = keras.layers.Dense(1, input_dim=self.mlp_hidden_size[-1])
        self.sigmoid = keras.activations.sigmoid
        self.loss = keras.losses.BinaryCrossentropy()

        self.optimizer = tf.optimizers.Adam(learning_rate)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        user, item = inputs
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        if self.is_mf_train:
            mf_output = user_mf_e * item_mf_e  # [batch_size, embedding_size]
        if self.is_mlp_train:
            mlp_output = self.mlp_layers(tf.concat([user_mlp_e, item_mlp_e], -1))  # [batch_size, layers[-1]]
        if self.is_mf_train and self.is_mlp_train:
            output = self.sigmoid(self.predict_layer(tf.concat([mf_output, mlp_output], -1)))
        elif self.is_mf_train:
            output = self.sigmoid(self.predict_layer(mf_output))
        elif self.is_mlp_train:
            output = self.sigmoid(self.predict_layer(mlp_output))
        else:
            raise RuntimeError('mf_train and mlp_train can not be False at the same time')
        return output

    @tf.function
    def train_step(self, batch):
        user, pos, label = batch
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
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        if self.is_mf_train:
            mf_output = user_mf_e * item_mf_e  # [batch_size, embedding_size]
        if self.is_mlp_train:
            mlp_output = self.mlp_layers(tf.concat([user_mlp_e, item_mlp_e], -1))  # [batch_size, layers[-1]]
        if self.is_mf_train and self.is_mlp_train:
            output = self.sigmoid(self.predict_layer(tf.concat([mf_output, mlp_output], -1)))
        elif self.is_mf_train:
            output = self.sigmoid(self.predict_layer(mf_output))
        elif self.is_mlp_train:
            output = self.sigmoid(self.predict_layer(mlp_output))
        else:
            raise RuntimeError('mf_train and mlp_train can not be False at the same time')
        return tf.squeeze(output)

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
