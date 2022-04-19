from abc import ABC

import tensorflow as tf
import numpy as np
import os
import random


class NARREModel(tf.keras.Model, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 l_w,
                 users_vocabulary_features,
                 items_vocabulary_features,
                 textual_words_feature_shape,
                 user_review_cnn_kernel,
                 user_review_cnn_features,
                 item_review_cnn_kernel,
                 item_review_cnn_features,
                 attention_size_user,
                 attention_size_item,
                 latent_size,
                 n_latent,
                 pretrained,
                 dropout_rate,
                 random_seed,
                 name="NARRE",
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
        self.l_w = l_w
        self.user_review_cnn_kernel = user_review_cnn_kernel
        self.user_review_cnn_features = user_review_cnn_features
        self.item_review_cnn_kernel = item_review_cnn_kernel
        self.item_review_cnn_features = item_review_cnn_features
        self.latent_size = latent_size
        self.dropout_rate = dropout_rate
        self.attention_size_user = attention_size_user
        self.attention_size_item = attention_size_item
        self.n_latent = n_latent
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

        # user and item embeddings
        self.iidW = tf.Variable(tf.initializers.random_uniform(-0.1, 0.1)(shape=[self.num_items + 2, self.latent_size]))
        self.uidW = tf.Variable(tf.initializers.random_uniform(-0.1, 0.1)(shape=[self.num_users + 2, self.latent_size]))
        self.iidmf = tf.Variable(
            tf.initializers.random_uniform(-0.1, 0.1)(shape=[self.num_items + 2, self.latent_size]))
        self.uidmf = tf.Variable(
            tf.initializers.random_uniform(-0.1, 0.1)(shape=[self.num_users + 2, self.latent_size]))

        # cnn
        self.user_convolutions = []
        for i, filter_size in enumerate(self.user_review_cnn_kernel):
            # Convolution Layer
            filter_shape = [filter_size, self.textual_words_feature_shape, 1, self.user_review_cnn_features]
            W = tf.Variable(tf.initializers.truncated_normal(stddev=0.1)(shape=filter_shape))
            b = tf.Variable(tf.constant(0.1, shape=[self.user_review_cnn_features]))
            self.user_convolutions.append((W, b))
        self.item_convolutions = []
        for i, filter_size in enumerate(self.item_review_cnn_kernel):
            # Convolution Layer
            filter_shape = [filter_size, self.textual_words_feature_shape, 1, self.item_review_cnn_features]
            W = tf.Variable(tf.initializers.truncated_normal(stddev=0.1)(shape=filter_shape))
            b = tf.Variable(tf.constant(0.1, shape=[self.item_review_cnn_features]))
            self.item_convolutions.append((W, b))

        # user and item attention
        self.num_filters_total_user = self.user_review_cnn_features * len(self.user_review_cnn_kernel)
        self.Wau = tf.Variable(
            tf.initializers.random_uniform(-0.1, 0.1)(shape=[self.num_filters_total_user, self.attention_size_user]))
        self.Wru = tf.Variable(
            tf.initializers.random_uniform(-0.1, 0.1)(shape=[self.latent_size, self.attention_size_user]))
        self.Wpu = tf.Variable(
            tf.initializers.random_uniform(-0.1, 0.1)(shape=[self.attention_size_user, 1]))
        self.bau = tf.Variable(tf.constant(0.1, shape=[self.attention_size_user]))
        self.bbu = tf.Variable(tf.constant(0.1, shape=[1]))
        self.num_filters_total_item = self.item_review_cnn_features * len(self.item_review_cnn_kernel)
        self.Wai = tf.Variable(
            tf.initializers.random_uniform(-0.1, 0.1)(shape=[self.num_filters_total_item, self.attention_size_item]))
        self.Wri = tf.Variable(
            tf.initializers.random_uniform(-0.1, 0.1)(shape=[self.latent_size, self.attention_size_item]))
        self.Wpi = tf.Variable(
            tf.initializers.random_uniform(-0.1, 0.1)(shape=[self.attention_size_item, 1]))
        self.bai = tf.Variable(tf.constant(0.1, shape=[self.attention_size_item]))
        self.bbi = tf.Variable(tf.constant(0.1, shape=[1]))

        # user and item dense
        self.dense_u = tf.keras.layers.Dense(units=self.n_latent,
                                             kernel_initializer=tf.initializers.RandomUniform(-0.1, 0.1),
                                             bias_initializer=tf.initializers.Constant(0.1))
        self.dense_i = tf.keras.layers.Dense(units=self.n_latent,
                                             kernel_initializer=tf.initializers.RandomUniform(-0.1, 0.1),
                                             bias_initializer=tf.initializers.Constant(0.1))

        # ncf
        self.Wmul = tf.Variable(tf.initializers.random_uniform(-0.1, 0.1)(shape=[self.n_latent, 1]))
        self.uidW2 = tf.Variable(tf.constant(0.1, shape=[self.num_users + 2]))
        self.iidW2 = tf.Variable(tf.constant(0.1, shape=[self.num_items + 2]))
        self.bised = tf.Variable(tf.constant(0.1))

        self.optimizer = tf.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    @tf.function
    def forward_user_embeddings(self, inputs, training=True):
        user, user_reviews, input_reuid = inputs
        review_num_u = user_reviews.shape[1]
        review_len_u = user_reviews.shape[2]
        embedded_user = tf.nn.embedding_lookup(self.W1, user_reviews)
        embedded_user = tf.expand_dims(embedded_user, -1)
        embedded_user = tf.reshape(embedded_user, [-1, review_len_u, self.textual_words_feature_shape, 1])

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
        h_pool_u = tf.concat(pooled_outputs_u, 3)
        h_pool_flat_u = tf.reshape(h_pool_u, [-1, review_num_u, self.num_filters_total_user])

        if training:
            h_drop_u = tf.nn.dropout(h_pool_flat_u, 0.0)
        else:
            h_drop_u = h_pool_flat_u

        iid_a = tf.nn.relu(tf.nn.embedding_lookup(self.iidW, input_reuid))
        u_j = tf.einsum('ajk,kl->ajl', tf.nn.relu(
            tf.einsum('ajk,kl->ajl', h_drop_u, self.Wau) + tf.einsum('ajk,kl->ajl', iid_a, self.Wru) + self.bau),
                        self.Wpu) + self.bbu
        u_a = tf.nn.softmax(u_j, 1)

        u_feas = tf.reduce_sum(tf.multiply(u_a, h_drop_u), 1)

        if training:
            u_feas = tf.nn.dropout(u_feas, self.dropout_rate)

        uid = tf.nn.embedding_lookup(self.uidmf, user)
        uid = tf.reshape(uid, [-1, self.latent_size])

        u_feas = self.dense_u(u_feas) + uid

        return u_feas

    @tf.function
    def forward_item_embeddings(self, inputs, training=True):
        item, item_reviews, input_reiid = inputs
        review_num_i = item_reviews.shape[1]
        review_len_i = item_reviews.shape[2]
        embedded_item = tf.nn.embedding_lookup(self.W2, item_reviews)
        embedded_item = tf.expand_dims(embedded_item, -1)
        embedded_item = tf.reshape(embedded_item, [-1, review_len_i, self.textual_words_feature_shape, 1])

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
        h_pool_i = tf.concat(pooled_outputs_i, 3)
        h_pool_flat_i = tf.reshape(h_pool_i, [-1, review_num_i, self.num_filters_total_item])

        if training:
            h_drop_i = tf.nn.dropout(h_pool_flat_i, 0.0)
        else:
            h_drop_i = h_pool_flat_i

        uid_a = tf.nn.relu(tf.nn.embedding_lookup(self.uidW, input_reiid))
        i_j = tf.einsum('ajk,kl->ajl', tf.nn.relu(
            tf.einsum('ajk,kl->ajl', h_drop_i, self.Wai) + tf.einsum('ajk,kl->ajl', uid_a, self.Wri) + self.bai),
                        self.Wpi) + self.bbi
        i_a = tf.nn.softmax(i_j, 1)

        i_feas = tf.reduce_sum(tf.multiply(i_a, h_drop_i), 1)

        if training:
            i_feas = tf.nn.dropout(i_feas, self.dropout_rate)

        iid = tf.nn.embedding_lookup(self.iidmf, item)
        iid = tf.reshape(iid, [-1, self.latent_size])

        i_feas = self.dense_i(i_feas) + iid

        return i_feas

    @tf.function
    def call(self, inputs, training=True):
        u_feas, u_bias, i_feas, i_bias = inputs
        FM = tf.multiply(u_feas, i_feas)
        FM = tf.nn.relu(FM)

        if training:
            FM = tf.nn.dropout(FM, self.dropout_rate)

        mul = tf.matmul(FM, self.Wmul)
        score = tf.reduce_sum(mul, 1)

        Feature_bias = u_bias + i_bias

        predictions = score + Feature_bias + self.bised

        return predictions

    @tf.function
    def predict(self, out_users, user, out_items, item, batch_user, batch_item):
        u_bias = tf.gather(self.uidW2, user)
        i_bias = tf.gather(self.iidW2, item)
        rui = self((out_users, u_bias, out_items, i_bias), training=False)
        return tf.reshape(rui, [batch_user, batch_item])

    @tf.function
    def train_step(self, batch):
        user, item, r, user_reviews, item_reviews, pos_user, pos_item = batch

        with tf.GradientTape() as t:
            u_feas = self.forward_user_embeddings((user, user_reviews, pos_user), training=True)
            i_feas = self.forward_item_embeddings((item, item_reviews, pos_item), training=True)
            u_bias = tf.gather(self.uidW2, user)
            i_bias = tf.gather(self.iidW2, item)
            xui = self(inputs=(u_feas, u_bias, i_feas, i_bias), training=True)

            loss = tf.reduce_mean(tf.abs(tf.subtract(xui, r)))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([
                tf.nn.l2_loss(self.Wau),
                tf.nn.l2_loss(self.Wai),
                tf.nn.l2_loss(self.Wru),
                tf.nn.l2_loss(self.Wri)
            ])

            # Loss to be optimized
            loss += reg_loss

        grads = t.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
