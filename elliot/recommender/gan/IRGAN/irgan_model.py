"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from elliot.dataset.samplers import pointwise_pos_neg_sampler as pws

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Generator(keras.Model):

    def __init__(self,
                 data,
                 factors=200,
                 learning_rate=0.001,
                 l_w=0, l_b=0, l_gan=0,
                 num_users=100,
                 num_items=100,
                 name="IRGAN-GEN",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self._factors = factors
        self._learning_rate = learning_rate
        self._l_w = l_w
        self._l_b = l_b
        self._l_gan = l_gan
        self._num_items = num_items
        self._num_users = num_users
        self.data = data

        self.initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=1234)

        self.sampler = pws.Sampler(self.data.i_train_dict)

        # Generator
        self.Bi = tf.Variable(tf.zeros(self._num_items), name='Bi_gen', dtype=tf.float32)
        self.Gu = tf.Variable(self.initializer(shape=[self._num_users, self._factors]), name='Gu_gen',
                              dtype=tf.float32)
        self.Gi = tf.Variable(self.initializer(shape=[self._num_items, self._factors]), name='Gi_gen',
                              dtype=tf.float32)

        self.optimizer = tf.optimizers.Adam(self._learning_rate)

    def call(self, inputs, training=None, mask=None):
        user, item = inputs
        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))

        xui = beta_i + tf.reduce_sum(gamma_u * gamma_i, 1)

        return xui, beta_i, gamma_u, gamma_i

    def train_step(self, batch):
        user, pos, label = batch

        with tf.GradientTape() as tape:
            # Clean Inference
            xui, beta_i, gamma_u, gamma_i = self(inputs=(user, pos), training=True)

            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(label, tf.float32), logits=xui)

            reg_loss = self._l_w * tf.nn.l2_loss(gamma_u) + tf.nn.l2_loss(gamma_i) + self._l_b * tf.nn.l2_loss(beta_i)

            loss += reg_loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    def train_step_with_reward(self, batch):
        user, pos, reward = batch

        with tf.GradientTape() as tape:
            # Clean Inference
            xui, beta_i, gamma_u, gamma_i = self(
                inputs=(tf.repeat(user[0], self._num_items), np.arange(self._num_items)), training=True)

            i_prob = tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(xui, [1, -1])), [-1]), pos)

            gan_loss = -tf.reduce_mean(tf.math.log(i_prob + 1e-5) * reward)

            reg_loss = self._l_w * tf.reduce_sum([tf.nn.l2_loss(tf.gather(gamma_u, user[0])),
                                                  tf.nn.l2_loss(tf.gather(gamma_i, pos))]) \
                       + self._l_b * tf.nn.l2_loss(tf.gather(beta_i, pos))

            gan_loss_reg = gan_loss + reg_loss

        grads = tape.gradient(gan_loss_reg, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return gan_loss_reg


class Discriminator(keras.Model):

    def __init__(self,
                 data,
                 factors=200,
                 learning_rate=0.001,
                 l_w=0, l_b=0, l_gan=0,
                 num_users=100,
                 num_items=100,
                 name="IRGAN-DIS",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self._factors = factors
        self._learning_rate = learning_rate
        self._l_w = l_w
        self._l_b = l_b
        self._l_gan = l_gan
        self._num_items = num_items
        self._num_users = num_users
        self.data = data

        self.initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=1234)

        self.sampler = pws.Sampler(self.data.i_train_dict)

        # Discriminator Model Parameters
        self.Bi = tf.Variable(tf.zeros(self._num_items), name='Bi_dis', dtype=tf.float32)
        self.Gu = tf.Variable(self.initializer(shape=[self._num_users, self._factors]), name='Gu_dis',
                              dtype=tf.float32)
        self.Gi = tf.Variable(self.initializer(shape=[self._num_items, self._factors]), name='Gi_dis',
                              dtype=tf.float32)

        self.optimizer = tf.optimizers.Adam(self._learning_rate)

    def call(self, inputs, training=None, mask=None):
        user, item = inputs
        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))

        xui = beta_i + tf.reduce_sum(gamma_u * gamma_i, 1)

        return xui, beta_i, gamma_u, gamma_i

    def train_step(self, batch):
        user, pos, label = batch

        with tf.GradientTape() as tape:
            # Clean Inference
            xui, beta_i, gamma_u, gamma_i = self(inputs=(user, pos), training=True)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(label, tf.float32), logits=xui)
            reg_loss = self._l_w * tf.nn.l2_loss(gamma_u) + self._l_w * tf.nn.l2_loss(
                gamma_i) + self._l_b * tf.nn.l2_loss(beta_i)

            loss += reg_loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return tf.reduce_sum(loss)


class IRGAN_model(keras.Model):

    def __init__(self,
                 predict_model,
                 data,
                 batch_size=512,
                 factors=200,
                 learning_rate=0.001,
                 l_w=0, l_b=0, l_gan=0,
                 num_users=100,
                 num_items=100,
                 g_pretrain_epochs=1,
                 d_pretrain_epochs=1,
                 g_epochs=5,
                 d_epochs=1,
                 sample_lambda=0.2,
                 random_seed=42,
                 name="IRGAN",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self._predict_model = predict_model
        self._data = data
        self._factors = factors
        self._learning_rate = learning_rate
        self._l_w = l_w
        self._l_b = l_b
        self._l_gan = l_gan
        self._num_items = num_items
        self._num_users = num_users
        self._g_pretrain_epochs = g_pretrain_epochs
        self._d_pretrain_epochs = d_pretrain_epochs
        self._g_epochs = g_epochs
        self._d_epochs = d_epochs
        self._batch_size = batch_size
        self._sample_lambda = sample_lambda

        self.initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=1234)

        # Discriminator
        self._discriminator = Discriminator(self._data,
                                            self._factors,
                                            self._learning_rate,
                                            self._l_w,
                                            self._l_b,
                                            self._l_gan,
                                            self._num_users,
                                            self._num_items)

        # Pretrain of D
        self.pre_train_discriminator()

        # Generator
        self._generator = Generator(self._data,
                                    self._factors,
                                    self._learning_rate,
                                    self._l_w,
                                    self._l_b,
                                    self._l_gan,
                                    self._num_users,
                                    self._num_items)

        # Pretrain of G
        self.pre_train_generator()

    def call(self, inputs, training=None):
        return self._generator(inputs) if self._predict_model == "generator" else self._discriminator(inputs)

    def train_step(self):
        for d_epoch in range(self._d_epochs):
            print(f'\n***** Train D - Epoch{d_epoch + 1}/{self._d_epochs}')
            dis_loss, step = 0, 0
            for batch in self._discriminator.sampler.step(self._discriminator.data.transactions, self._batch_size):
                dis_loss += self._discriminator.train_step(batch)
                step += 1
            dis_loss /= step

        for g_epoch in range(self._g_epochs):
            print(f'***** Train G - Epoch{g_epoch + 1}/{self._g_epochs}')
            gan_loss = 0
            for user in range(self._num_users):
                # print(f'***** Train G - Epoch{g_epoch+1}/{self._g_epochs} - User {user+1}/{self._num_users}')
                u, pos = self._data.sp_i_train_ratings.getrow(user).nonzero()
                pred_score, _, _, _ = self._generator(
                    inputs=(np.repeat(user, self._num_items), np.arange(self._num_items)))

                exp_pred_score = np.exp(pred_score / 0.5)
                prob = exp_pred_score / np.sum(exp_pred_score)

                # Here is the importance sampling.
                pn = (1 - self._sample_lambda) * prob
                pn[pos] += self._sample_lambda * 1.0 / len(pos)
                sample = np.random.choice(np.arange(self._num_items), 2 * len(pos), p=pn)

                # Get reward and adapt it with importance sampling
                reward_logits, _, _, _ = self._discriminator(inputs=(np.repeat(user, len(sample)), sample))
                reward = 2 * (tf.sigmoid(reward_logits) - 0.5)
                reward = reward * prob[sample] / pn[sample]

                # Update G
                gan_loss += self._generator.train_step_with_reward(batch=(np.repeat(user, len(sample)), sample, reward))
            gan_loss /= self._num_users

        return dis_loss, gan_loss

    def predict(self, start, stop, **kwargs):
        if self._predict_model == "generator":
            return self._generator.Bi + tf.matmul(self._generator.Gu[start:stop], self._generator.Gi, transpose_b=True)
        else:
            return self._discriminator.Bi + tf.matmul(self._discriminator.Gu[start:stop], self._discriminator.Gi,
                                                      transpose_b=True)

    def get_top_k(self, predictions, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, predictions, -np.inf), k=k, sorted=True)

    def get_positions(self, predictions, train_mask, items, inner_test_user_true_mask):
        predictions = tf.gather(predictions, inner_test_user_true_mask)
        train_mask = tf.gather(train_mask, inner_test_user_true_mask)
        equal = tf.reshape(items, [len(items), 1])
        i = tf.argsort(tf.where(train_mask, predictions, -np.inf), axis=-1,
                       direction='DESCENDING', stable=False, name=None)
        positions = tf.where(tf.equal(equal, i))[:, 1]
        return 1 - (positions / tf.reduce_sum(tf.cast(train_mask, tf.int64), axis=1))

    def pre_train_generator(self):
        for g_epoch in range(self._g_pretrain_epochs):
            for batch in self._generator.sampler.step(self._generator.data.transactions, self._batch_size):
                self._generator.train_step(batch)
            print(f'***** Pre Train G - Epoch {g_epoch + 1}/{self._g_pretrain_epochs}')

    def pre_train_discriminator(self):
        for d_epoch in range(self._d_pretrain_epochs):
            for batch in self._discriminator.sampler.step(self._discriminator.data.transactions, self._batch_size):
                self._discriminator.train_step(batch)
            print(f'***** Pre Train D - Epoch {d_epoch + 1}/{self._d_pretrain_epochs}\n')

    def get_config(self):
        raise NotImplementedError