"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AdversarialMatrixFactorizationModel(keras.Model):
    def __init__(self,
                 num_users,
                 num_items,
                 embed_mf_size,
                 lambda_weights,
                 learning_rate=0.01,
                 l_adv=1.0,
                 eps=0.5,
                 random_seed=42,
                 name="AdversarialMatrixFactorizationModel",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)
        self.num_users = num_users
        self.num_items = num_items
        self.embed_mf_size = embed_mf_size
        self.lambda_weights = lambda_weights
        self.l_adv = l_adv
        self.eps = eps

        self.initializer = tf.initializers.GlorotUniform()

        self.bias_item = tf.Variable(tf.zeros(self.num_items), name='bias_item', dtype=tf.float32)
        self.user_mf_embedding = tf.Variable(self.initializer(shape=[self.num_users, self.embed_mf_size]),
                                             name='item_mf_embedding', dtype=tf.float32)
        self.item_mf_embedding = tf.Variable(self.initializer(shape=[self.num_items, self.embed_mf_size]),
                                             name='item_mf_embedding', dtype=tf.float32)

        # Initialize the perturbation with 0 values
        self.delta_user_mf_embedding = tf.Variable(tf.zeros(shape=[self.num_users, self.embed_mf_size]),
                                                   dtype=tf.float32,
                                                   trainable=True)
        self.delta_item_mf_embedding = tf.Variable(tf.zeros(shape=[self.num_items, self.embed_mf_size]),
                                                   dtype=tf.float32,
                                                   trainable=True)

        self.loss = keras.losses.MeanSquaredError()
        self.optimizer = tf.optimizers.SGD(learning_rate)

    # @tf.function
    def call(self, inputs, adversarial=False, training=None, mask=None):
        user, item = inputs
        beta_i = tf.nn.embedding_lookup(self.bias_item, item)
        if adversarial:
            gamma_u = tf.nn.embedding_lookup(self.user_mf_embedding + self.delta_user_mf_embedding, user)
            gamma_i = tf.nn.embedding_lookup(self.item_mf_embedding + self.delta_item_mf_embedding, item)
        else:
            gamma_u = tf.nn.embedding_lookup(self.user_mf_embedding, user)
            gamma_i = tf.nn.embedding_lookup(self.item_mf_embedding, item)

        xui = beta_i + tf.reduce_sum(gamma_u * gamma_i, 1)

        return xui, beta_i, gamma_u, gamma_i

    # @tf.function
    def train_step(self, batch, user_adv_train=False):
        user, pos, label = batch
        with tf.GradientTape() as tape:
            # Clean Inference
            output, beta_i, gamma_u, gamma_i = self(inputs=(user, pos), training=True)
            loss = self.loss(label, output)

            if user_adv_train:
                # Build the Adversarial Perturbation on the Current Model Parameters
                self.build_perturbation(batch)

                # Clean Inference
                adversarial_output, beta_i, gamma_u, gamma_i = self(inputs=(user, pos), adversarial=True, training=True)

                adv_loss = self.loss(label, adversarial_output)

                loss += self.l_adv * adv_loss

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss

    # @tf.function
    def predict(self, inputs, training=False, **kwargs):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        output = self.call(inputs=inputs, training=training)
        return output

    # @tf.function
    def get_recs(self, inputs, training=False, **kwargs):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        user, item = inputs
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)

        mf_output = tf.reduce_sum(user_mf_e * item_mf_e, axis=-1)  # [batch_size, embedding_size]

        return tf.squeeze(mf_output)

    # @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)

    def build_perturbation(self, batch):
        """
        Evaluate Adversarial Perturbation with FGSM-like Approach
        """
        self.delta_user_mf_embedding = self.delta_user_mf_embedding * 0.0
        self.delta_item_mf_embedding = self.delta_item_mf_embedding * 0.0

        user, pos, label = batch
        with tf.GradientTape() as tape_adv:
            # Clean Inference
            adversarial_output, beta_i, gamma_u, gamma_i = self(inputs=(user, pos), adversarial=True, training=True)

            adv_loss = self.loss(label, adversarial_output)

        grad_user_mf_embedding, grad_item_mf_embedding = tape_adv.gradient(adv_loss, [self.user_mf_embedding, self.item_mf_embedding])
        grad_user_mf_embedding, grad_item_mf_embedding = tf.stop_gradient(grad_user_mf_embedding), tf.stop_gradient(grad_item_mf_embedding)
        self.delta_user_mf_embedding = tf.nn.l2_normalize(grad_user_mf_embedding, 1) * self.eps
        self.delta_item_mf_embedding = tf.nn.l2_normalize(grad_item_mf_embedding, 1) * self.eps
