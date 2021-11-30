import numpy as np

import tensorflow as tf
from tensorflow import keras


class CNN_module():

    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters, random_seed,
                 init_W=None, **kwargs):
        tf.random.set_seed(random_seed)
        self.max_len = max_len
        max_features = vocab_size
        vanila_dimension = 200
        projection_dimension = output_dimesion

        filter_lengths = [3, 4, 5]
        # self.model = keras.Sequential()

        '''Embedding Layer'''
        # self.model.add(keras.Input(name='input', shape=(max_len,)))
        inputs = keras.Input(name='input', shape=(max_len,))

        if init_W is None:
            # self.model.add(keras.layers.Embedding(max_features, emb_dim, input_length=max_len))
            embeddings = keras.layers.Embedding(max_features, emb_dim, input_length=max_len)(inputs)
        else:
            # self.model.add(keras.layers.Embedding(max_features, emb_dim, input_length=max_len, weights=[init_W / 20]))
            embeddings = keras.layers.Embedding(max_features, emb_dim, input_length=max_len, weights=[init_W / 20])(inputs)

        '''Convolution Layer & Max Pooling Layer'''

        convs = []
        for i in filter_lengths:
            # model_internal = keras.Sequential()
            # model_internal.add(keras.layers.Reshape((1, self.max_len, emb_dim), input_shape=(self.max_len, emb_dim)))
            # model_internal.add(keras.layers.Conv2D(nb_filters, i, emb_dim, activation="relu", padding='same'))
            # model_internal.add(keras.layers.MaxPooling2D(pool_size=(self.max_len - i + 1, 1), padding='same'))
            # model_internal.add(keras.layers.Flatten())
            reshape = keras.layers.Reshape((1, self.max_len, emb_dim), input_shape=(self.max_len, emb_dim))(embeddings)
            conv = keras.layers.Conv2D(nb_filters, i, emb_dim, activation="relu", padding='same')(reshape)
            max_pool = keras.layers.MaxPooling2D(pool_size=(self.max_len - i + 1, 1), padding='same')(conv)
            out = keras.layers.Flatten()(max_pool)

            convs.append(out)

        # concatenated = keras.layers.concatenate([c.output for c in convs], axis=-1)
        # model = keras.Model(inputs=[c.input for c in convs], outputs=concatenated)(embeddings)
        concat = keras.layers.Dense(vanila_dimension, activation='tanh')(keras.layers.concatenate(convs))
        '''Dropout Layer'''
        # self.model.add(keras.layers.Dense(vanila_dimension, activation='tanh'),
        #                     name='fully_connect', inputs=['unit_' + str(i) for i in filter_lengths])
        # self.model.add(keras.layers.Dropout(dropout_rate),
        #                     name='dropout', input='fully_connect')
        dropout = keras.layers.Dropout(dropout_rate)(concat)
        '''Projection Layer & Output Layer'''
        # self.model.add(keras.layers.Dense(projection_dimension, activation='tanh'),
        #                     name='projection', input='dropout')
        projection = keras.layers.Dense(projection_dimension, activation='tanh')(dropout)
        # Output Layer
        # self.model.add(keras.layers.Dense(name='output'))
        last = keras.layers.Dense(1, name='output')(projection)
        self.model = keras.Model(inputs=inputs, outputs=last)
        self.model.compile('rmsprop', {'output': 'mse'})

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def qualitative_CNN(self, vocab_size, emb_dim, max_len, nb_filters):
        self.max_len = max_len
        max_features = vocab_size

        filter_lengths = [3, 4, 5]
        print("Build model...")
        self.qual_model = keras.Sequential()
        self.qual_conv_set = {}
        '''Embedding Layer'''
        self.qual_model.add(keras.layers.Dense(
            name='input', input_dim=max_len))

        self.qual_model.add(keras.layers.Embedding(max_features, emb_dim, input_length=max_len,
                                           weights=self.model.nodes['sentence_embeddings'].get_weights()),
                                 name='sentence_embeddings', input='input')

        '''Convolution Layer & Max Pooling Layer'''
        for i in filter_lengths:
            model_internal = keras.Sequential()
            model_internal.add(
                keras.layers.Reshape(dims=(1, max_len, emb_dim), input_shape=(max_len, emb_dim)))
            self.qual_conv_set[i] = keras.layers.Convolution2D(nb_filters, i, emb_dim, activation="relu", weights=self.model.nodes[
                'unit_' + str(i)].layers[1].get_weights())
            model_internal.add(self.qual_conv_set[i])
            model_internal.add(keras.layers.MaxPooling2D(pool_size=(max_len - i + 1, 1)))
            model_internal.add(keras.layers.Flatten())

            self.qual_model.add(
                model_internal, name='unit_' + str(i), input='sentence_embeddings')
            self.qual_model.add(
                name='output_' + str(i), input='unit_' + str(i))

        self.qual_model.compile(
            'rmsprop', {'output_3': 'mse', 'output_4': 'mse', 'output_5': 'mse'})

    def train(self, X_train, V, item_weight, seed):
        X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=self.max_len)
        np.random.seed(seed)
        X_train = np.random.permutation(X_train)
        np.random.seed(seed)
        V = np.random.permutation(V)
        np.random.seed(seed)
        item_weight = np.random.permutation(item_weight)

        print("Train...CNN module")
        history = self.model.fit({'input': X_train, 'output': V},
                                 verbose=0, batch_size=self.batch_size, nb_epoch=self.nb_epoch,
                                 sample_weight={'output': item_weight})
        return history

    def get_projection_layer(self, X_train):
        X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=self.max_len)
        Y = self.model.predict({'input': X_train}, batch_size=len(X_train))
        return Y