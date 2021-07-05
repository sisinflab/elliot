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
                 learning_rate=0.001,
                 l_w=0, l_b=0, l_gan=0,
                 num_users=100,
                 num_items=100,
                 name="CFGAN-GEN",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self._learning_rate = learning_rate
        self._l_w = l_w
        self._l_b = l_b
        self._l_gan = l_gan
        self._num_items = num_items
        self._num_users = num_users
        self.data = data

        self.initializer = tf.initializers.GlorotUniform()

        self.sampler = pws.Sampler(self.data.i_train_dict)

        # Discriminator Model Parameters
        self.B = tf.Variable(tf.zeros(shape=[self._num_items]), name='B_gen', dtype=tf.float32)
        self.G = tf.Variable(self.initializer(shape=[self._num_items, self._num_items]), name='G_gen',
                             dtype=tf.float32)

        self.optimizer = tf.optimizers.Adam(self._learning_rate)

    def generate_fake_data(self, mask, C_u):
        r_hat = tf.nn.sigmoid(tf.matmul(tf.cast(C_u, tf.float32), self.G) + self.B)
        fake_data = tf.multiply(r_hat, mask)
        return fake_data

    def infer(self, C_u):
        r_hat = tf.nn.sigmoid(tf.matmul(C_u, self.G) + self.B)
        return r_hat

    def train_step(self, batch):
        C_u, mask, N_zr, g_sample, d_fake = batch

        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.math.log(1. - d_fake + 10e-5) + self._l_gan * tf.nn.l2_loss(
                tf.multiply(tf.cast(N_zr, tf.float32), g_sample)))

            reg_loss = self._l_w * tf.nn.l2_loss(self.G) + self._l_b * tf.nn.l2_loss(self.B)

            loss += reg_loss

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss


class Discriminator(keras.Model):

    def __init__(self,
                 data,
                 learning_rate=0.001,
                 l_w=0, l_b=0, l_gan=0,
                 num_users=100,
                 num_items=100,
                 name="CFGAN-DIS",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self._learning_rate = learning_rate
        self._l_w = l_w
        self._l_b = l_b
        self._l_gan = l_gan
        self._num_items = num_items
        self._num_users = num_users
        self.data = data

        self.initializer = tf.initializers.GlorotUniform()

        self.sampler = pws.Sampler(self.data.i_train_dict)

        # Discriminator Model Parameters
        self.B = tf.Variable(tf.zeros(shape=[1]), name='B_dis', dtype=tf.float32)
        self.G = tf.Variable(self.initializer(shape=[self._num_items * 2, 1]), name='G_dis',
                             dtype=tf.float32)

        self.optimizer = tf.optimizers.Adam(self._learning_rate)

    def discriminate_fake_data(self, X):
        disc_output = tf.nn.sigmoid(tf.matmul(tf.cast(X, tf.float32), self.G) + self.B)
        return disc_output

    def train_step(self, batch):
        C_u, mask, N_zr, g_sample = batch

        with tf.GradientTape() as tape:
            # Inference on the Discirminator
            disc_real = self.discriminate_fake_data(tf.concat([C_u, C_u], 1))
            disc_fake = self.discriminate_fake_data(tf.concat([g_sample, C_u], 1))

            loss = -tf.reduce_mean(tf.math.log(disc_real + 10e-5) + tf.math.log(1. - disc_fake + 10e-5))

            reg_loss = self._l_w * tf.nn.l2_loss(self.G) + self._l_b * tf.nn.l2_loss(self.B)

            loss += reg_loss

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss


class CFGAN_model(keras.Model):

    def __init__(self,
                 data,
                 batch_size=512,
                 learning_rate=0.001,
                 l_w=0, l_b=0, l_gan=0,
                 num_users=100,
                 num_items=100,
                 g_epochs=5,
                 d_epochs=1,
                 s_zr=0, s_pm=0,
                 random_seed=42,
                 name="CFGAN",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self._data = data
        self._learning_rate = learning_rate
        self._l_w = l_w
        self._l_b = l_b
        self._l_gan = l_gan
        self._num_items = num_items
        self._num_users = num_users
        self._g_epochs = g_epochs
        self._d_epochs = d_epochs
        self._batch_size = batch_size
        self._s_zr = s_zr
        self._s_pm = s_pm

        self.initializer = tf.initializers.GlorotUniform()

        # Discriminator
        self._discriminator = Discriminator(self._data,
                                            self._learning_rate,
                                            self._l_w,
                                            self._l_b,
                                            self._l_gan,
                                            self._num_users,
                                            self._num_items)

        # Generator
        self._generator = Generator(self._data,
                                    self._learning_rate,
                                    self._l_w,
                                    self._l_b,
                                    self._l_gan,
                                    self._num_users,
                                    self._num_items)

    def train_step(self, batch):
        C_u, mask, N_zr = batch

        gen_loss, dis_loss = 0, 0
        for g_epoch in range(self._g_epochs):  # G-Steps
            g_sample = self._generator.generate_fake_data(C_u, mask)
            d_fake = self._discriminator.discriminate_fake_data(tf.concat([g_sample, C_u], 1))
            gen_loss = self._generator.train_step(batch=(C_u, mask, N_zr, g_sample, d_fake))

        for d_epoch in range(self._d_epochs):  # D-Steps
            g_sample = self._generator.generate_fake_data(C_u, mask)
            dis_loss = self._discriminator.train_step(batch=(C_u, mask, N_zr, g_sample))

        return dis_loss, gen_loss

    def get_config(self):
        raise NotImplementedError

    def get_top_k(self, predictions, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, predictions, -np.inf), k=k, sorted=True)

    def predict(self, start, stop, **kwargs):
        vec = self._data.sp_i_train.tocsr()[start:stop].toarray()
        return self._generator.infer(vec)
