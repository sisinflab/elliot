"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class NPRModel(keras.Model):
    def __init__(self,
                 num_users,
                 num_items,
                 embed_mf_size, l_w, mlp_hidden_size, dropout, learning_rate=0.01,
                 random_seed=42,
                 name="NPR",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)
        self.num_users = num_users
        self.num_items = num_items
        self.embed_mf_size = embed_mf_size
        self.l_w = l_w
        self.mlp_hidden_size = mlp_hidden_size
        self.dropout = dropout

        self.initializer = tf.initializers.GlorotUniform()

        self.user_mf_embedding = keras.layers.Embedding(input_dim=self.num_users, output_dim=self.embed_mf_size,
                                                        embeddings_initializer=self.initializer, name='U_MF',
                                                        dtype=tf.float32)
        self.item_mf_embedding_1 = keras.layers.Embedding(input_dim=self.num_items, output_dim=self.embed_mf_size,
                                                          embeddings_initializer=self.initializer, name='I_MF_1',
                                                          dtype=tf.float32)
        self.item_mf_embedding_2 = keras.layers.Embedding(input_dim=self.num_items, output_dim=self.embed_mf_size,
                                                          embeddings_initializer=self.initializer, name='I_MF_2',
                                                          dtype=tf.float32)

        self.mlp_layers_1 = keras.Sequential()

        for units in mlp_hidden_size:
            # We can have a deeper MLP. In the paper is directly to 1
            self.mlp_layers_1.add(keras.layers.Dropout(dropout))
            self.mlp_layers_1.add(keras.layers.Dense(units, activation='relu'))

        self.mlp_layers_2 = keras.Sequential()

        for units in mlp_hidden_size:
            # We can have a deeper MLP. In the paper is directly to 1
            self.mlp_layers_2.add(keras.layers.Dropout(dropout))
            self.mlp_layers_2.add(keras.layers.Dense(units, activation='relu'))

        self.optimizer = tf.optimizers.Adam(learning_rate)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        user, item1, item2 = inputs
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e_1 = self.item_mf_embedding_1(item1)
        item_mf_e_2 = self.item_mf_embedding_2(item2)

        embedding_input_1 = user_mf_e * item_mf_e_1  # [batch_size, embedding_size]
        mlp_output_1 = self.mlp_layers_1(embedding_input_1)  # [batch_size, 1]

        embedding_input_2 = user_mf_e * item_mf_e_2
        mlp_output_2 = self.mlp_layers_2(embedding_input_2)  # [batch_size, 1]

        return tf.squeeze(mlp_output_1), tf.squeeze(mlp_output_2), user_mf_e, item_mf_e_1, item_mf_e_2

    #@tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            user, pos, neg = batch
            # Clean Inference
            mlp_output_1, mlp_output_2, user_mf_e, item_mf_e_1, item_mf_e_2 = self.call(inputs=(user, pos, neg),
                                                                                        training=True)

            difference = tf.clip_by_value(mlp_output_1 - mlp_output_2, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-difference))
            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(user_mf_e),
                                                 tf.nn.l2_loss(item_mf_e_1),
                                                 tf.nn.l2_loss(item_mf_e_2)])
            # Loss to be optimized
            loss += reg_loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss

    @tf.function
    def predict(self, inputs, training=False, **kwargs):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        u, i = inputs
        output_1, output_2, _, _, _ = self.call(inputs=(u, i, i), training=training)
        return (output_1 + output_2) * 0.5

    @tf.function
    def get_recs(self, inputs, training=False, **kwargs):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        user, item = inputs
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e_1 = self.item_mf_embedding_1(item)
        item_mf_e_2 = self.item_mf_embedding_2(item)

        mf_output_1 = user_mf_e * item_mf_e_1  # [batch_size, embedding_size]
        mf_output_2 = user_mf_e * item_mf_e_2  # [batch_size, embedding_size]

        mlp_output_1 = self.mlp_layers_1(mf_output_1)  # [batch_size, 1]
        mlp_output_2 = self.mlp_layers_2(mf_output_2)  # [batch_size, 1]

        return tf.squeeze((mlp_output_1+mlp_output_2)/2)

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
