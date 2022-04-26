from abc import ABC

import tensorflow as tf
import numpy as np
import os
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


class RMGModel(tf.keras.Model, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 word_cnn_fea_maps,
                 word_cnn_fea_kernel,
                 word_att,
                 sent_cnn_fea_maps,
                 sent_cnn_fea_kernel,
                 sent_att,
                 doc_att,
                 doc_att_u,
                 latent_size,
                 ui_att,
                 iu_att,
                 un_att,
                 in_att,
                 dropout_rate,
                 max_reviews_user,
                 max_reviews_item,
                 max_sents,
                 max_sent_length,
                 max_neighbor,
                 embed_vocabulary_features,
                 random_seed,
                 name="RMG",
                 **kwargs
                 ):
        super().__init__()

        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

        self.num_users = num_users
        self.num_items = num_items
        self.learning_rate = learning_rate
        self.word_cnn_fea_maps = word_cnn_fea_maps
        self.word_cnn_fea_kernel = word_cnn_fea_kernel
        self.word_att = word_att
        self.sent_cnn_fea_maps = sent_cnn_fea_maps
        self.sent_cnn_fea_kernel = sent_cnn_fea_kernel
        self.sent_att = sent_att
        self.doc_att = doc_att
        self.doc_att_u = doc_att_u
        self.latent_size = latent_size
        self.ui_att = ui_att
        self.iu_att = iu_att
        self.un_att = un_att
        self.in_att = in_att
        self.max_reviews_user = max_reviews_user
        self.max_reviews_item = max_reviews_item
        self.max_sents = max_sents
        self.max_sent_length = max_sent_length
        self.max_neighbor = max_neighbor
        self.embed_vocabulary_features = embed_vocabulary_features

        self.dropout_rate = dropout_rate

        sentence_input = Input(shape=(self.max_sent_length,), dtype='int32')
        embedding_layer = Embedding(self.embed_vocabulary_features.shape[0], self.embed_vocabulary_features.shape[1],
                                    weights=[self.embed_vocabulary_features], trainable=True)

        embedded_sequences = Dropout(self.dropout_rate)(embedding_layer(sentence_input))

        word_cnn_fea = Dropout(self.dropout_rate)(
            Convolution1D(filters=self.word_cnn_fea_maps,
                          kernel_size=self.word_cnn_fea_kernel,
                          padding='same',
                          activation='relu',
                          strides=1)(embedded_sequences))

        word_att = Dense(self.word_att, activation='tanh')(word_cnn_fea)
        word_att = Flatten()(Dense(1)(word_att))
        word_att = Activation('softmax')(word_att)
        sent_emb = Dot((1, 1))([word_cnn_fea, word_att])

        sent_encoder = Model([sentence_input], sent_emb)

        review_input = tf.keras.Input((self.max_sents, self.max_sent_length,), dtype='int32')

        review_encoder = TimeDistributed(sent_encoder)(review_input)

        sent_cnn_fea = Dropout(self.dropout_rate)(
            Convolution1D(filters=self.sent_cnn_fea_maps,
                          kernel_size=self.sent_cnn_fea_kernel,
                          padding='same',
                          activation='relu',
                          strides=1)(review_encoder))

        sent_att = Dense(self.sent_att, activation='tanh')(sent_cnn_fea)
        sent_att = Flatten()(Dense(1)(sent_att))
        word_att = Activation('softmax')(sent_att)
        doc_emb = tf.keras.layers.Dot((1, 1))([sent_cnn_fea, word_att])

        doc_encoder = Model([review_input], doc_emb)

        reviews_input_item = tf.keras.Input((self.max_reviews_item, self.max_sents, self.max_sent_length,),
                                            dtype='int32')
        reviews_input_user = tf.keras.Input((self.max_reviews_user, self.max_sents, self.max_sent_length,),
                                            dtype='int32')

        reviews_emb_item = TimeDistributed(doc_encoder)(reviews_input_item)
        reviews_emb_user = TimeDistributed(doc_encoder)(reviews_input_user)

        doc_att = Dense(self.doc_att, activation='tanh')(reviews_emb_item)
        doc_att = Flatten()(Dense(1)(doc_att))
        doc_att = Activation('softmax')(doc_att)
        item_emb = tf.keras.layers.Dot((1, 1))([reviews_emb_item, doc_att])

        doc_att_u = Dense(self.doc_att_u, activation='tanh')(reviews_emb_user)
        doc_att_u = Flatten()(Dense(1)(doc_att_u))
        doc_att_u = Activation('softmax')(doc_att_u)
        user_emb = tf.keras.layers.Dot((1, 1))([reviews_emb_user, doc_att_u])

        user_id = Input(shape=(1,), dtype='int32')
        item_id = Input(shape=(1,), dtype='int32')

        user_embedding = Embedding(self.num_users, self.latent_size, trainable=True)
        item_embedding = Embedding(self.num_items, self.latent_size, trainable=True)

        user_item_ids = tf.keras.Input((self.max_neighbor,), dtype='int32')
        item_user_ids = tf.keras.Input((self.max_neighbor,), dtype='int32')

        user_item_user_ids = tf.keras.Input((self.max_neighbor, self.max_neighbor), dtype='int32')
        item_user_item_ids = tf.keras.Input((self.max_neighbor, self.max_neighbor), dtype='int32')

        user_item_embedding = item_embedding(user_item_ids)
        item_user_embedding = user_embedding(item_user_ids)

        ui_att = Dense(self.ui_att, activation='tanh')(user_item_embedding)
        ui_att = Flatten()(Dense(1)(ui_att))
        ui_att = Activation('softmax')(ui_att)
        ui_emb = tf.keras.layers.Dot((1, 1))([user_item_embedding, ui_att])

        iu_att = Dense(self.iu_att, activation='tanh')(item_user_embedding)
        iu_att = Flatten()(Dense(1)(iu_att))
        iu_att_weight = Activation('softmax')(iu_att)
        iu_emb = tf.keras.layers.Dot((1, 1))([item_user_embedding, iu_att_weight])

        userencoder = Model([user_item_ids], ui_emb)
        itemencoder = Model([item_user_ids], iu_emb)

        user_encoder = TimeDistributed(itemencoder)(user_item_user_ids)
        item_encoder = TimeDistributed(userencoder)(item_user_item_ids)

        ufactor = concatenate([user_item_embedding, user_encoder])
        ifactor = concatenate([item_user_embedding, item_encoder])

        un_att = Dense(self.un_att, activation='tanh')(ufactor)
        un_att = Flatten()(Dense(1)(un_att))
        un_att = Activation('softmax')(un_att)
        user_emb_g = tf.keras.layers.Dot((1, 1))([ufactor, un_att])

        in_att = Dense(self.in_att, activation='tanh')(ifactor)
        in_att = Flatten()(Dense(1)(in_att))
        in_att = Activation('softmax')(in_att)
        item_emb_g = tf.keras.layers.Dot((1, 1))([ifactor, in_att])

        user_embedding = Flatten()(user_embedding(user_id))
        item_embedding = Flatten()(item_embedding(item_id))
        factor_u = concatenate([user_emb, user_embedding, user_emb_g])
        factor_i = concatenate([item_emb, item_embedding, item_emb_g])

        self.prediction_layer = Dense(1, activation='relu')

        self.model_user = Model([reviews_input_user, user_item_user_ids, user_item_ids, user_id], factor_u)
        self.model_item = Model([reviews_input_item, item_user_item_ids, item_user_ids, item_id], factor_i)

        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError()

    @tf.function
    def call(self, inputs, training=True):
        out_user, out_item = inputs
        return self.prediction_layer(tf.multiply(out_user, out_item))

    @tf.function
    def predict(self, inputs, batch_user, batch_item):
        rui = self(inputs, training=False)
        return tf.reshape(rui, [batch_user, batch_item])

    @tf.function
    def train_step(self, batch):
        inputs, r = batch
        with tf.GradientTape() as t:
            reviews_input_item, reviews_input_user, user_item_user_ids, \
            user_item_ids, item_user_item_ids, item_user_ids, item_id, user_id = inputs
            out_user = self.model_user([reviews_input_user, user_item_user_ids, user_item_ids, user_id],
                                       training=True)
            out_item = self.model_item([reviews_input_item, item_user_item_ids, item_user_ids, item_id],
                                       training=True)
            xui = self(inputs=[out_user, out_item])
            loss = self.loss(r, xui)

        grads = t.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
