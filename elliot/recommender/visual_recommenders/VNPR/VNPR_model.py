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


class VNPRModel(keras.Model):
    def __init__(self,
                 num_users,
                 num_items,
                 embed_mf_size, l_w, l_v, mlp_hidden_size, dropout, learning_rate=0.01,
                 num_image_feature=128,
                 random_seed=42,
                 name="VNPR",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)
        self.num_users = num_users
        self.num_items = num_items
        self.embed_mf_size = embed_mf_size
        self.num_image_feature = num_image_feature
        self.l_w = l_w
        self.l_v = l_v
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

        self.user_v_embedding = keras.layers.Embedding(input_dim=self.num_users, output_dim=self.num_image_feature,
                                                       embeddings_initializer=self.initializer, name='U_V',
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
        user, item1, feature_e_1, item2, feature_e_2 = inputs
        user_mf_e = self.user_mf_embedding(user)
        user_v_e = self.user_v_embedding(user)
        item_mf_e_1 = self.item_mf_embedding_1(item1)
        item_mf_e_2 = self.item_mf_embedding_2(item2)

        # embedding_input_1 = tf.concat([user_mf_e * item_mf_e_1, tf.tensordot(user_mf_e, self.user_visual_weight, axes=[[2], [0]]) * feature_e_1], axis=2)  # [batch_size, embedding_size]
        embedding_input_1 = tf.concat([user_mf_e * item_mf_e_1, user_v_e * feature_e_1], axis=1)
        mlp_output_1 = self.mlp_layers_1(embedding_input_1, training)  # [batch_size, 1]

        # embedding_input_2 = tf.concat([user_mf_e * item_mf_e_2, tf.tensordot(user_mf_e, self.user_visual_weight, axes=[[2], [0]]) * feature_e_2], axis=2)
        embedding_input_2 = tf.concat([user_mf_e * item_mf_e_2, user_v_e * feature_e_2], axis=1)
        mlp_output_2 = self.mlp_layers_2(embedding_input_2, training)  # [batch_size, 1]

        return tf.squeeze(mlp_output_1), tf.squeeze(mlp_output_2), user_mf_e, user_v_e, item_mf_e_1, item_mf_e_2

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            user, pos, feat_pos, neg, feat_neg = batch
            # Clean Inference
            mlp_output_1, mlp_output_2, user_mf_e, user_v_e, item_mf_e_1, item_mf_e_2 = self.call(
                inputs=(user, pos, feat_pos, neg, feat_neg),
                training=True)

            difference = tf.clip_by_value(mlp_output_1 - mlp_output_2, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-difference))

            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(user_mf_e),
                                                 tf.nn.l2_loss(item_mf_e_1),
                                                 tf.nn.l2_loss(item_mf_e_2)]) \
                       + self.l_v * tf.reduce_sum([tf.nn.l2_loss(user_v_e),
                                                   *[tf.nn.l2_loss(w1) for w1 in self.mlp_layers_1.trainable_variables],
                                                   *[tf.nn.l2_loss(w2) for w2 in
                                                     self.mlp_layers_2.trainable_variables]])
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
        feature_e = tf.nn.embedding_lookup(self.F, item)

        mf_output_1 = tf.concat([user_mf_e * item_mf_e_1, feature_e], axis=2)  # [batch_size, embedding_size]
        mf_output_2 = tf.concat([user_mf_e * item_mf_e_2, feature_e], axis=2)  # [batch_size, embedding_size]

        mlp_output_1 = self.mlp_layers_1(mf_output_1)  # [batch_size, 1]
        mlp_output_2 = self.mlp_layers_2(mf_output_2)  # [batch_size, 1]

        return tf.squeeze((mlp_output_1+mlp_output_2)/2)

    @tf.function
    def predict_item_batch(self, start, stop, item_mf_e_1, item_mf_e_2, feat):
        user_mf_e = self.user_mf_embedding(tf.range(start, stop))
        user_v_e = self.user_v_embedding(tf.range(start, stop))

        mf_output_1 = tf.concat([tf.expand_dims(user_mf_e, axis=1) * tf.expand_dims(item_mf_e_1, axis=0),
                                 tf.expand_dims(user_v_e, axis=1) * tf.expand_dims(feat, axis=0)], axis=2)
        mf_output_2 = tf.concat([tf.expand_dims(user_mf_e, axis=1) * tf.expand_dims(item_mf_e_2, axis=0),
                                 tf.expand_dims(user_v_e, axis=1) * tf.expand_dims(feat, axis=0)], axis=2)

        mlp_output_1 = self.mlp_layers_1(mf_output_1, training=False)  # [batch_size, batch_size_item, 1]
        mlp_output_2 = self.mlp_layers_2(mf_output_2, training=False)  # [batch_size, batch_size_item, 1]

        return tf.squeeze((mlp_output_1 + mlp_output_2) / 2, axis=2)

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
