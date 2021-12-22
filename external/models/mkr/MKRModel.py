"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Alberto Carlo Maria Mancino'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, alberto.mancino@poliba.it'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Layer, Input, Dropout


class MKRModel(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 learning_rate,
                 L1_flag,
                 l2_lambda,
                 embedding_size,
                 low_layers,
                 high_layers,
                 user_total,
                 item_total,
                 entity_total,
                 relation_total,
                 private_items_entitiesidx,
                 name="mkr",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(42)

        self.dropout_prob = 0.2

        self.learning_rate = learning_rate
        self.L1_flag = L1_flag
        self.l2_lambda = l2_lambda
        self.embedding_size = embedding_size
        self.rel_embedding_size = self.embedding_size
        self.L = low_layers
        self.H = high_layers
        self.user_total = user_total
        self.item_total = item_total
        self.ent_total = entity_total + 1
        self.rel_total = relation_total
        self.is_pretrained = False

        self.rs_loss = tf.keras.losses.BinaryCrossentropy()

        # store item to item-entity to (entity, item)
        self.private_items_entitiesidx = private_items_entitiesidx

        self.init_embeddings()
        self.init_MLPs()
        self.init_CrossCompress()

        keys, values = tuple(zip(*self.private_items_entitiesidx.items()))
        init = tf.lookup.KeyValueTensorInitializer(keys, values)
        self.paddingItems = tf.lookup.StaticHashTable(init, default_value=self.ent_total - 1)

        self.optimizer_rs = tf.optimizers.Adam(self.learning_rate)
        self.optimizer_kge = tf.optimizers.Adam(self.learning_rate)

    def init_embeddings(self):

        initializer = keras.initializers.GlorotNormal()

        # link prediction
        self.ent_embeddings = keras.layers.Embedding(input_dim=self.ent_total, output_dim=self.embedding_size,
                                                     embeddings_initializer=initializer,
                                                     trainable=False, dtype=tf.float32,
                                                     embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
        self.ent_embeddings(0)

        self.rel_embeddings = keras.layers.Embedding(input_dim=self.rel_total, output_dim=self.embedding_size,
                                                     embeddings_initializer=initializer,
                                                     trainable=False, dtype=tf.float32,
                                                     embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
        self.rel_embeddings(0)

        # recommender
        self.usr_embeddings = keras.layers.Embedding(input_dim=self.user_total, output_dim=self.embedding_size,
                                                     embeddings_initializer=initializer,
                                                     trainable=False, dtype=tf.float32,
                                                     embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
        self.usr_embeddings(0)

        self.itm_embeddings = keras.layers.Embedding(input_dim=self.item_total, output_dim=self.embedding_size,
                                                     embeddings_initializer=initializer,
                                                     trainable=False, dtype=tf.float32,
                                                     embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
        self.itm_embeddings(0)

    def init_MLPs(self):

        initializer = 'GlorotNormalV2'
        actv = 'sigmoid'

        # low depth MLPs
        self.user_mlp = Sequential()
        self.rel_mlp = Sequential()

        for _ in range(self.L):
            # user mlp
            self.user_mlp.add(Dropout(self.dropout_prob))
            self.user_mlp.add(Dense(self.embedding_size, activation=actv, kernel_initializer=initializer))

            # rel mlp
            self.rel_mlp.add(Dropout(self.dropout_prob))
            self.rel_mlp.add(Dense(self.embedding_size, activation=actv, kernel_initializer=initializer))

        self.user_mlp.build((None, self.embedding_size))
        self.rel_mlp.build((None, self.embedding_size))

        # high depth MLPs
        self.kge_mlp = Sequential()

        for _ in range(self.H - 1):
            # kge mlp
            self.rel_mlp.add(Dropout(self.dropout_prob))
            self.kge_mlp.add(Dense(self.embedding_size * 2, activation=actv, kernel_initializer=initializer))
        self.kge_mlp.add(Dense(self.embedding_size, activation=actv, kernel_initializer=initializer))
        self.kge_mlp.build((None, self.embedding_size * 2))

    def init_CrossCompress(self):
        emb = self.embedding_size

        item_input = Input((emb, 1))
        ent_input = Input((emb, 1))

        x = CrossCompress(self.embedding_size)([item_input, ent_input])
        for _ in range(self.L - 1):
            x = CrossCompress(self.embedding_size)(x)
        self.cc = keras.Model([item_input, ent_input], x)

    def get_config(self):
        raise NotImplementedError

    # @tf.function
    def call(self, inputs, **kwargs):

        score = 0

        if kwargs['is_rec']:
            u_ids, i_ids = inputs

            # lookup embeddings
            u_e = self.usr_embeddings(tf.squeeze(u_ids))
            i_e = self.itm_embeddings(tf.squeeze(i_ids))
            e_var = self.paddingItems.lookup(tf.squeeze(tf.cast(i_ids, tf.int32)))
            e_e = tf.expand_dims(self.ent_embeddings(e_var), axis=-1)

            # compute embeddings
            u_e = self.user_mlp(u_e)
            i_e, _ = self.cc([i_e, e_e])
            i_e = tf.squeeze(i_e)
            score = tf.math.sigmoid(tf.reduce_sum(u_e * i_e, axis=1))

            return score, u_e, i_e

        elif not kwargs['is_rec']:

            h, r, t, v, pn = inputs

            h_e = self.ent_embeddings(tf.squeeze(h))
            v_e = self.itm_embeddings(tf.squeeze(v))
            _, h_e = self.cc([v_e, h_e])
            h_e = tf.squeeze(h_e)

            r_e = self.rel_embeddings(tf.squeeze(r))
            r_e = self.rel_mlp(r_e)

            hr_e = tf.concat([h_e, r_e], axis=1)
            t_pred = self.kge_mlp(hr_e)

            t_e = self.ent_embeddings(tf.squeeze(t))
            score = tf.sigmoid(tf.reduce_sum((t_e * t_pred), axis=1)) * pn

        return score

    @tf.function
    def train_step_rec(self, batch, **kwargs):
        with tf.GradientTape() as tape:
            user, item, rating = batch
            score, u_e, i_e = self.call(inputs=(user, item), training=True, **kwargs)
            loss = self.rec_loss(rating, score, u_e, i_e)

        user_mlp_grads, cc_grads = tape.gradient(loss, [self.user_mlp.trainable_weights, self.cc.trainable_weights])
        user_mlp_grads, _ = tf.clip_by_global_norm(user_mlp_grads, 5)
        cc_grads, _ = tf.clip_by_global_norm(cc_grads, 5)
        self.optimizer_rs.apply_gradients(zip(user_mlp_grads, self.user_mlp.trainable_weights))
        self.optimizer_rs.apply_gradients(zip(cc_grads, self.cc.trainable_weights))
        return loss

    # @tf.function
    def train_step_kg(self, batch, **kwargs):
        with tf.GradientTape() as tape:
            scores = self.call(inputs=batch, training=True, **kwargs)
            loss = tf.reduce_sum(scores)

        rel_mlp_grads, kge_mlp_grads, cc_grads = tape.gradient(loss, [self.rel_mlp.trainable_weights,
                                                                      self.kge_mlp.trainable_weights,
                                                                      self.cc.trainable_weights])
        rel_mlp_grads, _ = tf.clip_by_global_norm(rel_mlp_grads, 5)
        kge_mlp_grads, _ = tf.clip_by_global_norm(kge_mlp_grads, 5)
        cc_grads, _ = tf.clip_by_global_norm(cc_grads, 5)

        self.optimizer_kge.apply_gradients(zip(rel_mlp_grads, self.rel_mlp.trainable_weights))
        self.optimizer_kge.apply_gradients(zip(kge_mlp_grads, self.kge_mlp.trainable_weights))
        self.optimizer_kge.apply_gradients(zip(cc_grads, self.cc.trainable_weights))
        return loss

    @tf.function
    def predict(self, inputs, training=False, **kwargs):
        score = self.call(inputs=inputs, training=training, is_rec=True)[0]
        return score

    @tf.function
    def get_recs(self, inputs, **kwargs):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        u_ids = inputs[0]
        i_ids = inputs[1]
        return self.call(inputs=(u_ids, i_ids), training=False, is_rec=True, **kwargs)[0]

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)

    def rec_loss(self, rating, score, usr_emb, itm_emb):
        return self.rs_loss(rating, score) + (tf.nn.l2_loss(usr_emb) + tf.nn.l2_loss(itm_emb))*self.l2_lambda

class CrossCompress(Layer):

    def __init__(self, embedding_dim, **kwargs):
        super(CrossCompress, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        w_initializer = tf.initializers.GlorotUniform()
        b_initializer = tf.initializers.Zeros()
        self.w_vv = tf.Variable(w_initializer(shape=(self.embedding_dim, 1)), trainable=True, name='w_vv')
        self.w_ve = tf.Variable(w_initializer(shape=(self.embedding_dim, 1)), trainable=True, name='w_ve')
        self.w_ev = tf.Variable(w_initializer(shape=(self.embedding_dim, 1)), trainable=True, name='w_ev')
        self.w_ee = tf.Variable(w_initializer(shape=(self.embedding_dim, 1)), trainable=True, name='w_ee')
        self.bias_v = tf.Variable(b_initializer(shape=(self.embedding_dim, 1)), trainable=True, name='b_v')
        self.bias_e = tf.Variable(b_initializer(shape=(self.embedding_dim, 1)), trainable=True, name='b_e')
        super(CrossCompress, self).build(input_shape)

    def call(self, input_data, **kwargs):
        item_embedding, entity_embedding = input_data
        entity_embedding = tf.transpose(entity_embedding, perm=[0, 2, 1])
        cross_matrix = tf.matmul(item_embedding, entity_embedding)
        cross_matrixT = tf.transpose(cross_matrix, perm=[0, 2, 1])
        item_embedding = tf.matmul(cross_matrix, self.w_vv) + tf.matmul(cross_matrixT, self.w_ev) + self.bias_v
        entity_embedding = tf.matmul(cross_matrix, self.w_ve) + tf.matmul(cross_matrixT, self.w_ee) + self.bias_e
        return [item_embedding, entity_embedding]

    def compute_output_shape(self, input_shape):
        return input_shape
