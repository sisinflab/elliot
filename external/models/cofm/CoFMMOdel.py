"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import tensorflow as tf
from tensorflow import keras


class cofm(keras.Model):

    def __init__(self,
                 learning_rate,
                 L1_flag,
                 l2_lambda,
                 norm_lambda,
                 isShare,
                 embedding_size,
                 user_total,
                 item_total,
                 entity_total,
                 relation_total,
                 new_map,
                 name="cofm",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(42)

        self.learning_rate = learning_rate
        self.L1_flag = L1_flag
        self.l2_lambda = l2_lambda
        self.is_share = isShare
        self.new_map = new_map
        self.embedding_size = embedding_size
        self.user_total = user_total
        self.item_total = item_total
        # padding when item are not aligned with any entity
        self.ent_total = entity_total
        self.rel_total = relation_total
        self.is_pretrained = False
        self.norm_lambda = norm_lambda

        initializer = keras.initializers.GlorotNormal()
        # FM
        self.user_embeddings = keras.layers.Embedding(input_dim=self.user_total, output_dim=self.embedding_size,
                                                      embeddings_initializer=initializer,
                                                      trainable=True, dtype=tf.float32,
                                                      embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
        self.user_embeddings(0)
        self.user_embeddings.weights[0] = tf.math.l2_normalize(self.user_embeddings.weights[0])

        self.user_bias = keras.layers.Embedding(input_dim=self.user_total, output_dim=1,
                                                embeddings_initializer=tf.keras.initializers.Zeros(),
                                                trainable=True, dtype=tf.float32,
                                                embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
        self.user_bias(0)

        self.item_bias = keras.layers.Embedding(input_dim=self.item_total, output_dim=1,
                                                embeddings_initializer=tf.keras.initializers.Zeros(),
                                                trainable=True, dtype=tf.float32,
                                                embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
        self.item_bias(0)

        self.bias = tf.Variable([0.])

        # trans-E

        self.ent_embeddings = keras.layers.Embedding(input_dim=self.ent_total, output_dim=self.embedding_size,
                                                     embeddings_initializer=initializer,
                                                     trainable=True, dtype=tf.float32,
                                                     embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
        self.ent_embeddings(0)
        self.ent_embeddings.weights[0] = tf.math.l2_normalize(self.ent_embeddings.weights[0])

        self.rel_embeddings = keras.layers.Embedding(input_dim=self.rel_total, output_dim=self.embedding_size,
                                                     embeddings_initializer=initializer,
                                                     trainable=True, dtype=tf.float32,
                                                     embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
        self.rel_embeddings(0)
        self.rel_embeddings.weights[0] = tf.math.l2_normalize(self.rel_embeddings.weights[0])

        if self.is_share:
            assert self.item_total == self.ent_total, "item numbers didn't match entities!"
            self.item_embeddings = self.ent_embeddings
        else:
            self.item_embeddings = keras.layers.Embedding(input_dim=self.item_total, output_dim=self.embedding_size,
                                                          embeddings_initializer=initializer,
                                                          trainable=True, dtype=tf.float32,
                                                          embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
            self.item_embeddings(0)
            self.item_embeddings.weights[0] = tf.math.l2_normalize(self.item_embeddings.weights[0])

        keys, values = tuple(zip(*self.new_map.items()))
        initMappingItems = tf.lookup.KeyValueTensorInitializer(keys, values)
        initMappingEntities = tf.lookup.KeyValueTensorInitializer(values, keys)
        self.mappingItems = tf.lookup.StaticHashTable(initMappingItems, default_value=-1)
        self.mappingEntities = tf.lookup.StaticHashTable(initMappingEntities, default_value=-1)
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.one = tf.Variable(1.0)
        self.zero = tf.Variable(0.0)

        def get_config(self):
            raise NotImplementedError

    # @tf.function
    def call(self, inputs, training=None, **kwargs):

        if kwargs['is_rec']:
            u_ids, i_ids = inputs
            batch_size = len(u_ids)

            u_e = self.user_embeddings(tf.squeeze(u_ids))
            i_e = self.item_embeddings(tf.squeeze(i_ids))
            u_b = self.user_bias(tf.squeeze(u_ids))
            i_b = self.item_bias(tf.squeeze(i_ids))

            score = self.bias + u_b + i_b + tf.squeeze(tf.matmul(tf.expand_dims(u_e, len(tf.shape(u_e))-1),
                                                                 tf.expand_dims(i_e, len(tf.shape(i_e)))), axis=-1)

        elif not kwargs['is_rec']:
            h, t, r = inputs
            h_e = self.ent_embeddings(h)
            t_e = self.ent_embeddings(t)
            r_e = self.rel_embeddings(r)

            if self.L1_flag:
                score = tf.reduce_sum(tf.abs(h_e + r_e - t_e), -1)
            else:
                score = tf.reduce_sum((h_e + r_e - t_e) ** 2, -1)

        return score

    @tf.function
    def train_step_rec(self, batch, **kwargs):

        with tf.GradientTape() as tape:
            user, pos, neg = batch

            i_ids = tf.squeeze((tf.concat([pos, neg], 0)))
            e_ids = self.mappingItems.lookup(tf.squeeze(tf.cast(i_ids, tf.int32)))
            indices = ~tf.equal(e_ids, -1)

            i_ids = tf.boolean_mask(i_ids, indices)
            e_ids = tf.boolean_mask(e_ids, indices)

            pos_score = self.call(inputs=(user, pos), training=True, **kwargs)
            neg_score = self.call(inputs=(user, neg), training=True, **kwargs)

            losses = self.bprLoss(pos_score, neg_score)

            if not self.is_share:
                ent_embeddings = self.ent_embeddings(e_ids)
                item_embeddings = self.item_embeddings(i_ids)
                losses += self.norm_lambda * self.pNormLoss(ent_embeddings, item_embeddings, L1_flag=self.L1_flag)

        grads = tape.gradient(losses, self.trainable_weights)
        grads, _ = tf.clip_by_global_norm(grads, 5)  # fix clipping value
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return losses

    @tf.function
    def train_step_kg(self, batch, **kwargs):
        with tf.GradientTape() as tape:
            ph, pr, pt, nh, nr, nt = batch

            e_ids = tf.squeeze((tf.concat([ph, pt, nh, nt], 0)))
            i_ids = self.mappingEntities.lookup(tf.squeeze(tf.cast(e_ids, tf.int32)))
            indices = ~tf.equal(i_ids, -1)

            i_ids = tf.boolean_mask(i_ids, indices)
            e_ids = tf.boolean_mask(e_ids, indices)

            pos_score = self.call(inputs=(ph, pt, pr), training=True, **kwargs)
            neg_score = self.call(inputs=(nh, nt, nr), training=True, **kwargs)

            losses = self.marginLoss(pos_score, neg_score, 1)  # fix margin loss value
            ent_embeddings = self.ent_embeddings(tf.concat([ph, pt, nh, nt], 0))
            rel_embeddings = self.rel_embeddings(tf.concat([pr, nr], 0))
            losses += self.normLoss(ent_embeddings) + self.normLoss(rel_embeddings)
            losses = kwargs['kg_lambda'] * losses
            if not self.is_share:
                ent_embeddings = self.ent_embeddings(e_ids)
                item_embeddings = self.item_embeddings(i_ids)
                losses += self.norm_lambda * self.pNormLoss(ent_embeddings, item_embeddings, L1_flag=self.L1_flag)

        grads = tape.gradient(losses, self.trainable_weights)
        grads, _ = tf.clip_by_global_norm(grads, 1)  # fix clipping value
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return losses

    @tf.function
    def predict(self, inputs, training=False, **kwargs):
        score = self.call(inputs=inputs, training=training, is_rec=True)
        return score

    @tf.function
    def get_recs(self, inputs, training=False, **kwargs):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        u_ids, i_ids = inputs

        score = self.call(inputs=(u_ids, i_ids), training=False, is_rec=True, **kwargs)

        # e_var = self.paddingItems.lookup(tf.squeeze(tf.cast(i_ids, tf.int32)))
        # u_e = self.user_embeddings(u_ids)
        # i_e = self.item_embeddings(i_ids)
        # e_e = self.ent_embeddings(e_var)
        # ie_e = i_e + e_e
        #
        # _, r_e, norm = self.getPreferences(u_e, ie_e)
        #
        # proj_u_e = self.projection_trans_h(u_e, norm)
        # proj_i_e = self.projection_trans_h(ie_e, norm)
        #
        # if self.L1_flag:
        #     score = tf.reduce_sum(tf.abs(proj_u_e + r_e - proj_i_e), -1)
        # else:
        #     score = tf.reduce_sum((proj_u_e + r_e - proj_i_e) ** 2, -1)
        return tf.squeeze(score)

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)

    @tf.function
    def pNormLoss(self, emb1, emb2, L1_flag=False):
        if L1_flag:
            distance = tf.reduce_sum(tf.abs(emb1 - emb2), 1)
        else:
            distance = tf.reduce_sum((emb1 - emb2) ** 2, 1)
        return tf.reduce_mean(distance)

    @tf.function
    def bprLoss(self, pos, neg, target=1.0):
        loss = - tf.math.log_sigmoid(target * (pos - neg))
        return tf.reduce_mean(loss)

    @tf.function
    def marginLoss(self, pos, neg, margin):
        zero_tensor = tf.zeros(len(pos))
        return tf.reduce_sum(tf.math.maximum(pos - neg + margin, zero_tensor))

    @tf.function
    def normLoss(self, embeddings, dim=-1):
        norm = tf.reduce_sum(embeddings ** 2, axis=dim, keepdims=True)
        return tf.reduce_sum(tf.math.maximum(norm - self.one, self.zero))