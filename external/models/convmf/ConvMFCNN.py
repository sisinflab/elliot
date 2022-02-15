import numpy as np

import tensorflow as tf
from tensorflow import keras
import os
import random


class CNN_module():

    def __init__(self, output_dimesion, vocab_size, dropout_rate, batch_size, emb_dim, max_len, nb_filters, init_W,
                 random_seed, **kwargs):

        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = 5
        max_features = vocab_size
        vanila_dimension = 200
        projection_dimension = output_dimesion

        filter_lengths = [3, 4, 5]

        inputs = keras.Input(name='input', shape=(max_len,))

        if init_W is None:
            embeddings = keras.layers.Embedding(max_features, emb_dim, input_length=max_len)(inputs)
        else:
            embeddings = keras.layers.Embedding(max_features, emb_dim, input_length=max_len, weights=[init_W / 20])(
                inputs)

        convs = []
        for i in filter_lengths:
            reshape = keras.layers.Reshape((1, self.max_len, emb_dim), input_shape=(self.max_len, emb_dim))(embeddings)
            conv = keras.layers.Conv2D(nb_filters, i, emb_dim, activation="relu", padding='same')(reshape)
            max_pool = keras.layers.MaxPooling2D(pool_size=(self.max_len - i + 1, 1), padding='same')(conv)
            out = keras.layers.Flatten()(max_pool)

            convs.append(out)

        concat = keras.layers.Dense(vanila_dimension, activation='tanh')(keras.layers.concatenate(convs))
        dropout = keras.layers.Dropout(dropout_rate)(concat)
        projection = keras.layers.Dense(projection_dimension, activation='tanh', name='output')(dropout)
        # last = keras.layers.Dense(1, name='output')(projection)
        self.model = keras.Model(inputs=inputs, outputs=projection)
        self.model.compile('rmsprop', {'output': 'mse'})

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def train(self, X_train, V, item_weight, seed):
        X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=self.max_len)
        np.random.seed(seed)
        X_train = np.random.permutation(X_train)
        np.random.seed(seed)
        V = np.random.permutation(V)
        np.random.seed(seed)
        item_weight = np.random.permutation(item_weight)

        print("Train...CNN module")
        history = self.model.fit(X_train, V, verbose=0, batch_size=self.batch_size, epochs=self.epochs,
                                 sample_weight={'output': item_weight})
        return history

    def get_projection_layer(self, X_train):
        X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=self.max_len)
        Y = self.model.predict({'input': X_train}, batch_size=len(X_train))
        return Y
