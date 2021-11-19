"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import tensorflow as tf
from tensorflow import keras


class jtup(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 learning_rate,
                 L1_flag,
                 l2_lambda,
                 embedding_size,
                 user_total,
                 item_total,
                 entity_total,
                 relation_total,
                 new_map,
                 name="jtup",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(42)

        self.learning_rate = learning_rate
        self.L1_flag = L1_flag
        self.l2_lambda = l2_lambda
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
        # self.i_map = i_map
        # store item to item-entity to (entity, item)
        self.new_map = new_map
        # todo: simiplifying the init

        initializer = keras.initializers.GlorotNormal()
        # transup
        self.user_embeddings = keras.layers.Embedding(input_dim=self.user_total, output_dim=self.embedding_size,
                                                      embeddings_initializer=initializer,
                                                      trainable=True, dtype=tf.float32,
                                                      embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
        self.user_embeddings(0)
        self.user_embeddings.weights[0] = tf.math.l2_normalize(self.user_embeddings.weights[0])

        self.item_embeddings = keras.layers.Embedding(input_dim=self.item_total, output_dim=self.embedding_size,
                                                      embeddings_initializer=initializer,
                                                      trainable=True, dtype=tf.float32,
                                                      embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
        self.item_embeddings(0)
        self.item_embeddings.weights[0] = tf.math.l2_normalize(self.item_embeddings.weights[0])

        self.pref_embeddings = keras.layers.Embedding(input_dim=self.rel_total, output_dim=self.embedding_size,
                                                      embeddings_initializer=initializer,
                                                      trainable=True, dtype=tf.float32,
                                                      embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
        self.pref_embeddings(0)
        self.pref_embeddings.weights[0] = tf.math.l2_normalize(self.pref_embeddings.weights[0])

        self.pref_norm_embeddings = keras.layers.Embedding(input_dim=self.rel_total, output_dim=self.embedding_size,
                                                           embeddings_initializer=initializer,
                                                           trainable=True, dtype=tf.float32,
                                                           embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
        self.pref_norm_embeddings(0)
        self.pref_norm_embeddings.weights[0] = tf.math.l2_normalize(self.pref_norm_embeddings.weights[0])

        # transh

        ent_embeddings_w = tf.Variable(initial_value=initializer(shape=(self.ent_total, self.embedding_size)))
        ent_embeddings_w = tf.concat([tf.math.l2_normalize(ent_embeddings_w[:-1, :]), tf.zeros([1, self.embedding_size])], 0)

        self.ent_embeddings = keras.layers.Embedding(input_dim=self.ent_total, output_dim=self.embedding_size,
                                                     weights=[ent_embeddings_w],
                                                     trainable=True, dtype=tf.float32,
                                                     embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
        self.ent_embeddings(0)

        self.rel_embeddings = keras.layers.Embedding(input_dim=self.rel_total, output_dim=self.embedding_size,
                                                     embeddings_initializer=keras.initializers.GlorotNormal(),
                                                     trainable=True, dtype=tf.float32,
                                                     embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
        self.rel_embeddings(0)
        self.rel_embeddings.weights[0] = tf.math.l2_normalize(self.rel_embeddings.weights[0])

        self.norm_embeddings = keras.layers.Embedding(input_dim=self.rel_total, output_dim=self.embedding_size,
                                                      embeddings_initializer=keras.initializers.GlorotNormal(),
                                                      trainable=True, dtype=tf.float32,
                                                      embeddings_regularizer=keras.regularizers.l2(self.l2_lambda))
        self.norm_embeddings(0)
        self.norm_embeddings.weights[0] = tf.math.l2_normalize(self.norm_embeddings.weights[0])

        keys, values = tuple(zip(*self.new_map.items()))
        init = tf.lookup.KeyValueTensorInitializer(keys, values)
        self.paddingItems = tf.lookup.StaticHashTable(
            init,
            default_value=self.ent_total-1)
        # self.optimizer = tfa.optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=)
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.one = tf.Variable(1.0)
        self.zero = tf.Variable(0.0)

    def get_config(self):
        raise NotImplementedError

    # def convert_function(self, row):
    #     return tf.math.reduce_mean(self.item_embedding.weights[0][row > 0], axis=0)

    # def paddingItems(self, i_ids):
    #     # i_ids, pad_index = input
    #     padded_e_ids = []
    #     pad_index = self.ent_total-1
    #     for i_id in i_ids:
    #         # new_index = self.i_map[i_id]
    #         # ent_id = self.new_map[new_index][0]
    #         ent_id = self.new_map[int(i_id)]
    #         padded_e_ids.append([ent_id] if ent_id != -1 else [pad_index])
    #     return np.array(padded_e_ids)

    @tf.function
    def call(self, inputs, training=None, **kwargs):

        if kwargs['is_rec']:
            u_ids, i_ids = inputs
            e_var = self.paddingItems.lookup(tf.squeeze(tf.cast(i_ids, tf.int32)))
            u_e = self.user_embeddings(tf.squeeze(u_ids))
            i_e = self.item_embeddings(tf.squeeze(i_ids))
            e_e = self.ent_embeddings(tf.squeeze(e_var))
            ie_e = i_e + e_e

            _, r_e, norm = self.getPreferences(u_e, ie_e)

            proj_u_e = self.projection_trans_h(u_e, norm)
            proj_i_e = self.projection_trans_h(ie_e, norm)

            if self.L1_flag:
                score = tf.reduce_sum(tf.abs(proj_u_e + r_e - proj_i_e), -1)
            else:
                score = tf.reduce_sum((proj_u_e + r_e - proj_i_e) ** 2, -1)

        elif not kwargs['is_rec']:
            h, t, r = inputs
            h_e = self.ent_embeddings(h)
            t_e = self.ent_embeddings(t)
            r_e = self.rel_embeddings(r)
            norm_e = self.norm_embeddings(r)

            proj_h_e = self.projection_trans_h(h_e, norm_e)
            proj_t_e = self.projection_trans_h(t_e, norm_e)

            if self.L1_flag:
                score = tf.reduce_sum(tf.abs(proj_h_e + r_e - proj_t_e), -1)
            else:
                score = tf.reduce_sum((proj_h_e + r_e - proj_t_e) ** 2, -1)

        return score

    def getPreferences(self, u_e, i_e, use_st_gumbel=False):
        # use item and user embedding to compute preference distribution
        # pre_probs: batch * rel, or batch * item * rel
        pre_probs = tf.matmul(u_e + i_e, tf.transpose(self.pref_embeddings.weights[0] + self.rel_embeddings.weights[0])) / 2
        if use_st_gumbel:
            pre_probs = self.st_gumbel_softmax(pre_probs)

        r_e = tf.matmul(pre_probs, self.pref_embeddings.weights[0] + self.rel_embeddings.weights[0]) / 2
        norm = tf.matmul(pre_probs, self.pref_norm_embeddings.weights[0] + self.norm_embeddings.weights[0]) / 2

        return pre_probs, r_e, norm

    def projection_trans_h(self, original, norm):
        return original - tf.reduce_sum(original * norm, axis=len(original.shape.as_list()) - 1, keepdims=True) * norm

    def st_gumbel_softmax(self, logits, temperature=1.0):
        eps = 1e-20
        u = tf.random.uniform(tf.shape(logits), dtype=tf.float32)
        gumbel_noise = -tf.math.log(-tf.math.log(u + eps) + eps)
        y = logits + gumbel_noise
        y = self.masked_softmax(logits=y / temperature)
        y_argmax = tf.argmax(y, axis=len(y.shape) - 1)
        y_hard = self.convert_to_one_hot(
            indices=y_argmax,
            num_classes=tf.shape(y)[len(y.shape) - 1])

        # gumbel_softmax_distribution_ = GumbelSoftmax(1.0, logits=logits, dtype=tf.float32)
        # one_hot = gumbel_softmax_distribution_.convert_to_one_hot(logits)

        y = tf.stop_gradient(y_hard - y) + y
        return y

    def masked_softmax(self, logits):
        eps = 1e-20
        probs = tf.nn.softmax(logits, axis=len(logits.shape) - 1)
        return probs

    def convert_to_one_hot(self, indices, num_classes):
        one_hot = tf.one_hot(indices, num_classes)
        return one_hot

    @tf.function
    def train_step_rec(self, batch, **kwargs):

        with tf.GradientTape() as tape:

            user, pos, neg = batch

            pos_score = self.call(inputs=(user, pos), training=True, **kwargs)
            neg_score = self.call(inputs=(user, neg), training=True, **kwargs)

            losses = self.bprLoss(pos_score, neg_score)
            losses += self.orthogonalLoss(self.pref_embeddings.weights[0], self.pref_norm_embeddings.weights[0])

        grads = tape.gradient(losses, self.trainable_weights)
        grads, _ = tf.clip_by_global_norm(grads, 5) # fix clipping value
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return losses

    @tf.function
    def train_step_kg(self, batch, **kwargs):
        with tf.GradientTape() as tape:

            ph, pr, pt, nh, nr, nt = batch

            pos_score = self.call(inputs=(ph, pt, pr), training=True, **kwargs)
            neg_score = self.call(inputs=(nh, nt, nr), training=True, **kwargs)

            losses = self.marginLoss(pos_score, neg_score, 1) # fix margin loss value
            ent_embeddings = self.ent_embeddings(tf.concat([ph, pt, nh, nt], 0))
            rel_embeddings = self.rel_embeddings(tf.concat([pr, nr], 0))
            norm_embeddings = self.norm_embeddings(tf.concat([pr, nr], 0))
            losses += self.orthogonalLoss(rel_embeddings, norm_embeddings)
            losses += self.normLoss(ent_embeddings) + self.normLoss(rel_embeddings)
            losses = kwargs['kg_lambda'] * losses

        grads = tape.gradient(losses, self.trainable_weights)
        grads, _ = tf.clip_by_global_norm(grads, 1) # fix clipping value
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
    def bprLoss(self, pos, neg, target=1.0):
        loss = - tf.math.log_sigmoid(target * (pos - neg))
        return tf.reduce_mean(loss)

    @tf.function
    def orthogonalLoss(self, rel_embeddings, norm_embeddings):
        return tf.reduce_sum(
            tf.reduce_sum(norm_embeddings * rel_embeddings, axis=-1, keepdims=True) ** 2 /
            tf.reduce_sum(rel_embeddings ** 2, axis=-1, keepdims=True))

    @tf.function
    def normLoss(self, embeddings, dim=-1):
        norm = tf.reduce_sum(embeddings ** 2, axis=dim, keepdims=True)
        return tf.reduce_sum(tf.math.maximum(norm - self.one, self.zero))

    @tf.function
    def marginLoss(self, pos, neg, margin):
        zero_tensor = tf.zeros(len(pos))
        return tf.reduce_sum(tf.math.maximum(pos - neg + margin, zero_tensor))
