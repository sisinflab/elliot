"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Antonio Ferrara'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it,' \
            'daniele.malitesta@poliba.it, antonio.ferrara@poliba.it'

import os
from typing import Union, Text

import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class FactorizationMachineModel(keras.Model):
    def __init__(self,
                 num_users,
                 num_items,
                 num_features,
                 factors,
                 lambda_weights,
                 learning_rate=0.01,
                 random_seed=42,
                 name="FM",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self.factors = factors
        self.lambda_weights = lambda_weights

        self.initializer = tf.initializers.GlorotUniform()

        if self.num_features:
            self.factorization = FactorizationMachineLayer(field_dims=[self.num_users, self.num_items, self.num_features],
                                                           factors=self.factors, kernel_initializer=self.initializer,
                                                           kernel_regularizer=keras.regularizers.l2(self.lambda_weights))
        else:
            self.factorization = MatrixFactorizationLayer(num_users=self.num_users,num_items=num_items,
                                                          factors=self.factors, kernel_initializer=self.initializer,
                                                          kernel_regularizer=keras.regularizers.l2(self.lambda_weights))

        self.loss = keras.losses.MeanSquaredError()

        self.optimizer = tf.optimizers.Adam(learning_rate)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        transaction = inputs

        return self.factorization(inputs=transaction, training=training)

    @tf.function
    def train_step(self, batch):
        transaction, label = batch
        with tf.GradientTape() as tape:
            # Clean Inference
            output = self.factorization(inputs=transaction, training=True)
            loss = self.loss(label, output)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss

    @tf.function
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
        if self.num_features:
            output = tf.map_fn(lambda row: self.call(inputs=row, training=training),
                               tf.convert_to_tensor(inputs))
        else:
            output = self.call(inputs=inputs, training=training)

        return tf.squeeze(output)

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)

############################## Linear ####################

@tf.keras.utils.register_keras_serializable()
class Linear(tf.keras.layers.Layer):
    def __init__(self,
                 field_dims,
                 kernel_initializer: Union[Text, tf.keras.initializers.Initializer] = "truncated_normal",
                 **kwargs):
        super().__init__(**kwargs)
        self._field_dims = np.sum(field_dims)
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._supports_masking = True

        self._field_embedding = keras.layers.Embedding(input_dim=self._field_dims, output_dim=1,
                                                       embeddings_initializer=self._kernel_initializer, name='Bias',
                                                       dtype=tf.float32)
        self._g_bias = tf.Variable(0., name='GlobalBias')

        # Force initialization
        self._field_embedding(0)

        self.built = True

    @tf.function
    def call(self, x0: tf.Tensor, training=None) -> tf.Tensor:
        x = tf.map_fn(lambda row: tf.math.reduce_sum(self._field_embedding.weights[0][row>0], axis=0), x0)
        return self._g_bias + x

    @tf.function
    def get_config(self):
        config = {
            "kernel_initializer":
                tf.keras.initializers.serialize(self._kernel_initializer)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

##############################

############################## Embeddings ####################


@tf.keras.utils.register_keras_serializable()
class Embedding(tf.keras.layers.Layer):
    def __init__(
            self,
            field_dims,
            factors,
            kernel_initializer: Union[
                Text, tf.keras.initializers.Initializer] = "truncated_normal",
            kernel_regularizer: Union[Text, None,
                                      tf.keras.regularizers.Regularizer] = None,
            **kwargs):

        super().__init__(**kwargs)

        self._field_dims = np.sum(field_dims)
        self._factors = factors
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

        self._supports_masking = True
        self._embedding = keras.layers.Embedding(input_dim=self._field_dims, output_dim=self._factors,
                                                 embeddings_initializer=self._kernel_initializer, name='Embedding',
                                                 embeddings_regularizer=self._kernel_regularizer,
                                                 dtype=tf.float32)

        # Force initialization
        self._embedding(0)
        self.built = True

    @tf.function
    def call(self, x0: tf.Tensor, training=None) -> tf.Tensor:
        return tf.map_fn(lambda row: (tf.reduce_sum(
            tf.matmul(self._embedding.weights[0][row>0], tf.transpose(self._embedding.weights[0][row > 0])),
            axis=(-2, -1)) - tf.reduce_sum( self._embedding.weights[0][row>0] ** 2, axis=(-2,-1))) * 0.5, x0)

    @tf.function
    def get_config(self):
        config = {
            "kernel_initializer":
                tf.keras.initializers.serialize(self._kernel_initializer),
            "kernel_regularizer":
                tf.keras.regularizers.serialize(self._kernel_regularizer),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

##############################

############################## FM Layer ####################


@tf.keras.utils.register_keras_serializable()
class FactorizationMachineLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            field_dims,
            factors,
            kernel_initializer: Union[
                Text, tf.keras.initializers.Initializer] = "truncated_normal",
            kernel_regularizer: Union[Text, None,
                                      tf.keras.regularizers.Regularizer] = None,
            **kwargs):

        super().__init__(**kwargs)

        self.embedding = Embedding(field_dims, factors, kernel_initializer, kernel_regularizer)
        self.linear = Linear(field_dims, tf.initializers.zeros())

        self._supports_masking = True

    @tf.function
    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        linear = self.linear(inputs, training)
        second_order = tf.expand_dims(self.embedding(inputs, training), axis=-1)
        return linear + second_order

    @tf.function
    def get_config(self):
        config = {
            "use_bias":
                self._use_bias,
            "kernel_initializer":
                tf.keras.initializers.serialize(self._kernel_initializer),
            "bias_initializer":
                tf.keras.initializers.serialize(self._bias_initializer),
            "kernel_regularizer":
                tf.keras.regularizers.serialize(self._kernel_regularizer),
            "bias_regularizer":
                tf.keras.regularizers.serialize(self._bias_regularizer),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable()
class MatrixFactorizationLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            num_users,
            num_items,
            factors,
            kernel_initializer: Union[
                Text, tf.keras.initializers.Initializer] = "truncated_normal",
            kernel_regularizer: Union[Text, None,
                                      tf.keras.regularizers.Regularizer] = None,
            **kwargs):

        super().__init__(**kwargs)

        self.num_users = num_users
        self.num_items = num_items

        self.user_mf_embedding = keras.layers.Embedding(input_dim=num_users, output_dim=factors,
                                                        embeddings_initializer=kernel_initializer, name='U_MF',
                                                        embeddings_regularizer=kernel_regularizer,
                                                        dtype=tf.float32)
        self.item_mf_embedding = keras.layers.Embedding(input_dim=num_items, output_dim=factors,
                                                        embeddings_regularizer=kernel_regularizer,
                                                        embeddings_initializer=kernel_initializer, name='I_MF',
                                                        dtype=tf.float32)

        self.u_bias = keras.layers.Embedding(input_dim=num_users, output_dim=1,
                                             embeddings_initializer=tf.initializers.zeros(), name='B_U_MF',
                                             dtype=tf.float32)
        self.i_bias = keras.layers.Embedding(input_dim=num_items, output_dim=1,
                                             embeddings_initializer=tf.initializers.zeros(), name='B_I_MF',
                                             dtype=tf.float32)

        self.bias_ = tf.Variable(0., name='GB')

        self.user_mf_embedding(0)
        self.item_mf_embedding(0)
        self.u_bias(0)
        self.i_bias(0)

        self._supports_masking = True

    @tf.function
    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        user, item = inputs
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        mf_output = tf.reduce_sum(user_mf_e * item_mf_e, axis=-1)

        return mf_output + self.bias_ + tf.squeeze(self.u_bias(user), axis=-1) + tf.squeeze(self.i_bias(item), axis=-1)

    @tf.function
    def get_config(self):
        config = {
            "use_bias":
                self._use_bias,
            "kernel_initializer":
                tf.keras.initializers.serialize(self._kernel_initializer),
            "bias_initializer":
                tf.keras.initializers.serialize(self._bias_initializer),
            "kernel_regularizer":
                tf.keras.regularizers.serialize(self._kernel_regularizer),
            "bias_regularizer":
                tf.keras.regularizers.serialize(self._bias_regularizer),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
