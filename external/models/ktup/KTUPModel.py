"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import logging
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class jtup(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 L1_flag,
                 embedding_size,
                 user_total,
                 item_total,
                 entity_total,
                 relation_total,
                 i_map,
                 new_map,
                 name="jtup",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(42)

        self.L1_flag = L1_flag
        # self.is_share = isShare
        # self.use_st_gumbel = use_st_gumbel
        self.embedding_size = embedding_size
        self.user_total = user_total
        self.item_total = item_total
        # padding when item are not aligned with any entity
        self.ent_total = entity_total + 1
        self.rel_total = relation_total
        self.is_pretrained = False
        # store item to item-entity dic
        self.i_map = i_map
        # store item-entity to (entity, item)
        self.new_map = new_map
        # todo: simiplifying the init

        initializer = keras.initializers.GlorotNormal()
        # transup
        self.user_embeddings = keras.layers.Embedding(input_dim=self.user_total, output_dim=self.embedding_size,
                                                     embeddings_initializer=initializer,
                                                     trainable=True, dtype=tf.float32)
        self.user_embeddings(0)
        self.user_embeddings.weights[0] = tf.math.l2_normalize(self.item_embedding.weights[0])

        self.item_embeddings = keras.layers.Embedding(input_dim=self.item_total, output_dim=self.embedding_size,
                                                      embeddings_initializer=initializer,
                                                      trainable=True, dtype=tf.float32)
        self.item_embeddings(0)
        self.item_embeddings.weights[0] = tf.math.l2_normalize(self.item_embeddings.weights[0])

        self.pref_embeddings = keras.layers.Embedding(input_dim=self.rel_total, output_dim=self.embedding_size,
                                                      embeddings_initializer=initializer,
                                                      trainable=True, dtype=tf.float32)
        self.pref_embeddings(0)
        self.pref_embeddings.weights[0] = tf.math.l2_normalize(self.pref_embeddings.weights[0])

        self.pref_norm_embeddings = keras.layers.Embedding(input_dim=self.rel_total, output_dim=self.embedding_size,
                                                      embeddings_initializer=initializer,
                                                      trainable=True, dtype=tf.float32)
        self.pref_norm_embeddings(0)
        self.pref_norm_embeddings.weights[0] = tf.math.l2_normalize(self.pref_norm_embeddings.weights[0])

        # transh

        ent_embeddings_w = tf.Variable(initial_value=initializer(self.ent_total, self.embedding_size))
        ent_embeddings_w = tf.concat([tf.math.l2_normalize(ent_embeddings_w[:-1, :]), tf.zeros([1, 5])], 0)

        self.ent_embeddings = keras.layers.Embedding(input_dim=self.ent_total, output_dim=self.embedding_size,
                                                           weights=[ent_embeddings_w],
                                                           trainable=True, dtype=tf.float32)
        self.ent_embeddings(0)

        self.rel_embeddings = keras.layers.Embedding(input_dim=self.rel_total, output_dim=self.embedding_size,
                                                     embeddings_initializer=keras.initializers.GlorotNormal(),
                                                     trainable=True, dtype=tf.float32)
        self.rel_embeddings(0)
        self.rel_embeddings.weights[0] = tf.math.l2_normalize(self.rel_embeddings.weights[0])

        self.norm_embeddings = keras.layers.Embedding(input_dim=self.rel_total, output_dim=self.embedding_size,
                                                     embeddings_initializer=keras.initializers.GlorotNormal(),
                                                     trainable=True, dtype=tf.float32)
        self.norm_embeddings(0)
        self.norm_embeddings.weights[0] = tf.math.l2_normalize(self.norm_embeddings.weights[0])

        # self.item_embedding = keras.layers.Embedding(input_dim=item_factors.shape[0], output_dim=item_factors.shape[1],
        #                                              weights=[item_factors],
        #                                              embeddings_regularizer=keras.regularizers.l2(regularization_lambda),
        #                                              trainable=True, dtype=tf.float32)
        # # needed for initialization
        # self.item_embedding(0)
        # self.encoder = Encoder(latent_dim=latent_dim,
        #                        intermediate_dim=intermediate_dim,
        #                        dropout_rate=dropout_rate,
        #                        regularization_lambda=regularization_lambda)
        # self.decoder = Decoder(output_dim,
        #                        intermediate_dim=intermediate_dim,
        #                        regularization_lambda=regularization_lambda)
        # self.optimizer = tf.optimizers.Adam(learning_rate)
        # tf.Graph().finalize()

    def get_config(self):
        raise NotImplementedError

    def convert_function(self, row):
        return tf.math.reduce_mean(self.item_embedding.weights[0][row > 0], axis=0)

    def paddingItems(self, i_ids, pad_index):
        padded_e_ids = []
        for i_id in i_ids:
            new_index = self.i_map[i_id]
            ent_id = self.new_map[new_index][0]
            padded_e_ids.append(ent_id if ent_id != -1 else pad_index)
        return padded_e_ids

    # @tf.function
    def call(self, inputs, training=None, **kwargs):

        if kwargs['is_rec']:
            u_ids, i_ids = inputs
            e_var = self.paddingItems(i_ids.data, self.ent_total-1)
            u_e = self.user_embeddings(u_ids)
            i_e = self.item_embeddings(i_ids)
            e_e = self.ent_embeddings(e_var)
            ie_e = i_e + e_e

            _, r_e, norm = self.getPreferences(u_e, ie_e)

            proj_u_e = self.projection_trans_h(u_e, norm)
            proj_i_e = self.projection_trans_h(ie_e, norm)

            if self.L1_flag:
                score = tf.reduce_sum(tf.abs(proj_u_e + r_e - proj_i_e), 1)
            else:
                score = tf.reduce_sum((proj_u_e + r_e - proj_i_e) ** 2, 1)

        elif not kwargs['is_rec']:
            h, t, r = inputs
            h_e = self.ent_embeddings(h)
            t_e = self.ent_embeddings(t)
            r_e = self.rel_embeddings(r)
            norm_e = self.norm_embeddings(r)

            proj_h_e = self.projection_trans_h(h_e, norm_e)
            proj_t_e = self.projection_trans_h(t_e, norm_e)

            if self.L1_flag:
                score = tf.reduce_sum(tf.abs(proj_h_e + r_e - proj_t_e), 1)
            else:
                score = tf.reduce_sum((proj_h_e + r_e - proj_t_e) ** 2, 1)

        return score

    def getPreferences(self, u_e, i_e):
        # use item and user embedding to compute preference distribution
        # pre_probs: batch * rel, or batch * item * rel
        pre_probs = tf.matmul(u_e + i_e, tf.transpose(self.pref_embeddings.weight + self.rel_embeddings.weight)) / 2

        r_e = tf.matmul(pre_probs, self.pref_embeddings.weight + self.rel_embeddings.weight) / 2
        norm = tf.matmul(pre_probs, self.pref_norm_embeddings.weight + self.norm_embeddings.weight) / 2

        return pre_probs, r_e, norm

    def projection_trans_h(self, original, norm):
        return original - tf.reduce_sum(original * norm, dim=len(original.size()) - 1, keepdim=True) * norm

    # @tf.function
    def train_step_rec(self, batch, **kwargs):
        with tf.GradientTape() as tape:

            user, pos, neg = batch

            pos_score = self.call(inputs=(user, pos), training=True, **kwargs)
            neg_score = self.call(inputs=(user, neg), training=True, **kwargs)

            losses = self.bprLoss(pos_score, neg_score)
            losses += self.orthogonalLoss(self.pref_embeddings.weight, self.pref_norm_embeddings.weight)

        gradients, variables = zip(*self.optimizer.compute_gradients(losses))
        gradients, _ = tf.clip_by_global_norm(gradients, 1) # fix clipping value
        self.optimizer.apply_gradients(zip(tape.gradient(gradients, self.trainable_weights), self.trainable_weights))

        return losses

    # @tf.function
    def train_step_kg(self, batch, **kwargs):
        with tf.GradientTape() as tape:

            ph, pt, pr, nh, nt, nr = batch

            pos_score = self.call(inputs=(ph, pt, pr), training=True, **kwargs)
            neg_score = self.call(inputs=(nh, nt, nr), training=True, **kwargs)

            losses = self.marginLoss(pos_score, neg_score, 0.01) # fix margin loss value
            ent_embeddings = self.ent_embeddings(tf.concat([ph, pt, nh, nt]))
            rel_embeddings = self.rel_embeddings(tf.concat([pr, nr]))
            norm_embeddings = self.norm_embeddings(tf.concat([pr, nr]))
            losses += self.orthogonalLoss(rel_embeddings, norm_embeddings)
            losses += self.normLoss(ent_embeddings) + self.normLoss(rel_embeddings)
            losses = kwargs['kg_lambda'] * losses

        gradients, variables = zip(*self.optimizer.compute_gradients(losses))
        gradients, _ = tf.clip_by_global_norm(gradients, 1) # fix clipping value
        self.optimizer.apply_gradients(zip(tape.gradient(gradients, self.trainable_weights), self.trainable_weights))

        return losses

    @tf.function
    def predict(self, inputs, training=False, **kwargs):
        score = self.call(inputs=inputs, training=training, is_rec=True)
        return score

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)

    def bprLoss(self, pos, neg, target=1.0):
        loss = - tf.math.log_sigmoid(target * (pos - neg))
        return loss.mean()

    def orthogonalLoss(self, rel_embeddings, norm_embeddings):
        return tf.reduce_sum(
            tf.reduce_sum(norm_embeddings * rel_embeddings, dim=1, keepdim=True) ** 2 /
            tf.reduce_sum(rel_embeddings ** 2, dim=1, keepdim=True))

    def normLoss(self, embeddings, dim=1):
        norm = tf.reduce_sum(embeddings ** 2, dim=dim, keepdim=True)
        return tf.reduce_sum(tf.reduce_max(norm - (tf.Variable(tf.Tensor([1.0]))), (tf.Variable(tf.Tensor([0.0])))))

    def marginLoss(self, pos, neg, margin):
        zero_tensor = tf.zeros(len(pos))
        return tf.reduce_sum(tf.math.maximum(pos - neg + margin, zero_tensor))
