from abc import ABC

import tensorflow as tf
import numpy as np
import os
import random


class DeepCoNNModel(tf.keras.Model, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 users_vocabulary_features,
                 items_vocabulary_features,
                 textual_words_feature_shape,
                 user_review_cnn_kernel,
                 user_review_cnn_features,
                 item_review_cnn_kernel,
                 item_review_cnn_features,
                 latent_size,
                 fm_k,
                 dropout_rate,
                 pretrained,
                 random_seed,
                 name="DeepCoNN",
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
        self.user_review_cnn_kernel = user_review_cnn_kernel
        self.user_review_cnn_features = user_review_cnn_features
        self.item_review_cnn_kernel = item_review_cnn_kernel
        self.item_review_cnn_features = item_review_cnn_features
        self.latent_size = latent_size
        self.fm_k = fm_k
        self.dropout_rate = dropout_rate
        self.pretrained = pretrained

        # user and item vocabulary
        if self.pretrained:
            self.W1 = tf.Variable(tf.convert_to_tensor(users_vocabulary_features, dtype=tf.float32))
            self.W2 = tf.Variable(tf.convert_to_tensor(items_vocabulary_features, dtype=tf.float32))
        else:
            self.W1 = tf.Variable(
                tf.initializers.random_uniform(-0.1, 0.1)(shape=[users_vocabulary_features.shape[0], 300]))
            self.W2 = tf.Variable(
                tf.initializers.random_uniform(-0.1, 0.1)(shape=[items_vocabulary_features.shape[0], 300]))

        self.textual_words_feature_shape = textual_words_feature_shape

        # cnn
        self.user_convolutions = []
        for i, filter_size in enumerate(self.user_review_cnn_kernel):
            self.user_convolutions.append((
                tf.Variable(initial_value=tf.random.truncated_normal([filter_size,
                                                                      self.textual_words_feature_shape,
                                                                      1,
                                                                      self.user_review_cnn_features], stddev=0.1)),
                tf.Variable(initial_value=tf.constant(0.1, shape=[self.user_review_cnn_features]))))
        self.item_convolutions = []
        for i, filter_size in enumerate(self.item_review_cnn_kernel):
            self.item_convolutions.append((
                tf.Variable(initial_value=tf.random.truncated_normal([filter_size,
                                                                      self.textual_words_feature_shape,
                                                                      1,
                                                                      self.item_review_cnn_features], stddev=0.1)),
                tf.Variable(initial_value=tf.constant(0.1, shape=[self.item_review_cnn_features]))))

        # user and item dense
        self.num_filters_total_user = self.user_review_cnn_features * len(self.user_review_cnn_kernel)
        self.dense_u = tf.keras.layers.Dense(units=self.latent_size,
                                             kernel_initializer=tf.keras.initializers.glorot_normal(),
                                             bias_initializer=tf.initializers.Constant(0.1))
        self.num_filters_total_item = self.item_review_cnn_features * len(self.item_review_cnn_kernel)
        self.dense_i = tf.keras.layers.Dense(units=self.latent_size,
                                             kernel_initializer=tf.keras.initializers.glorot_normal(),
                                             bias_initializer=tf.initializers.Constant(0.1))

        # parameter for FM
        self.WF1 = tf.Variable(tf.random.uniform(minval=-0.1, maxval=0.1, shape=[self.latent_size * 2, 1]))
        self.WF2 = tf.Variable(tf.random.uniform(minval=-0.1, maxval=0.1, shape=[self.latent_size * 2, self.fm_k]))
        self.B = tf.Variable(tf.constant(0.1))

        self.optimizer = tf.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    @tf.function
    def forward_user_embeddings(self, inputs, training=True):
        user, user_reviews = inputs
        review_len_u = user_reviews.shape[1]

        embedded_user = tf.nn.embedding_lookup(self.W1, user_reviews)
        embedded_user = tf.expand_dims(embedded_user, -1)

        pooled_outputs_u = []

        for i, filter_size in enumerate(self.user_review_cnn_kernel):
            conv = tf.nn.conv2d(
                embedded_user,
                self.user_convolutions[i][0],
                strides=[1, 1, 1, 1],
                padding="VALID")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, self.user_convolutions[i][1]))
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, review_len_u - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID')
            pooled_outputs_u.append(pooled)
            
        h_pool_u = tf.concat(pooled_outputs_u, axis=1)
        h_pool_flat_u = tf.reshape(h_pool_u, [-1, self.num_filters_total_user])

        u_fea = self.dense_u(h_pool_flat_u)
        
        if training:
            u_fea = tf.nn.dropout(u_fea, self.dropout_rate)

        return u_fea

    @tf.function
    def forward_item_embeddings(self, inputs, training=True):
        item, item_reviews = inputs
        review_len_i = item_reviews.shape[1]

        embedded_item = tf.nn.embedding_lookup(self.W2, item_reviews)
        embedded_item = tf.expand_dims(embedded_item, -1)

        pooled_outputs_i = []

        for i, filter_size in enumerate(self.item_review_cnn_kernel):
            conv = tf.nn.conv2d(
                embedded_item,
                self.item_convolutions[i][0],
                strides=[1, 1, 1, 1],
                padding="VALID")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, self.item_convolutions[i][1]))
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, review_len_i - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID')
            pooled_outputs_i.append(pooled)
        h_pool_i = tf.concat(pooled_outputs_i, axis=1)
        h_pool_flat_i = tf.reshape(h_pool_i, [-1, self.num_filters_total_item])

        i_fea = self.dense_i(h_pool_flat_i)
        
        if training:
            i_fea = tf.nn.dropout(i_fea, self.dropout_rate)

        return i_fea

    @tf.function
    def call(self, inputs, training=True):
        u_feas, i_feas = inputs
        z = tf.nn.relu(tf.concat([u_feas, i_feas], -1))

        one = tf.matmul(z, self.WF1)

        inte1 = tf.matmul(z, self.WF2)
        inte2 = tf.matmul(tf.square(z), tf.square(self.WF2))

        inter = (tf.square(inte1) - inte2) * 0.5

        if training:
            inter = tf.nn.dropout(inter, self.dropout_rate)
        
        inter = tf.reduce_sum(inter, -1, keepdims=True)
        
        predictions = tf.squeeze(one + inter + self.B)
        return tf.math.sigmoid(predictions)

    @tf.function
    def predict(self, out_users, out_items, batch_user, batch_item):
        rui = tf.math.sigmoid(self((out_users, out_items), training=False))
        return tf.reshape(rui, [batch_user, batch_item])

    @tf.function
    def train_step(self, batch):
        #user, item, r, user_reviews, item_reviews = batch
        user, pos, neg, user_reviews, item_reviews_pos, item_reviews_neg = batch
        with tf.GradientTape() as t:
            u_feas = self.forward_user_embeddings((user, user_reviews), training=True)
            i_pos_feas = self.forward_item_embeddings((pos, item_reviews_pos), training=True)
            i_neg_feas = self.forward_item_embeddings((neg, item_reviews_neg), training=True)
            xu_pos = self(inputs=(u_feas, i_pos_feas), training=True)
            xu_neg = self(inputs=(u_feas, i_neg_feas), training=True)
            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))
            
        grads = t.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
