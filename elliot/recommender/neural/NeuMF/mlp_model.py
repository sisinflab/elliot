"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, Dropout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(0)

tf.random.set_seed(0)


class MLPModel(keras.Model):

    def __init__(self, input_shape, layers, dropout=0., activation='relu', **kwargs):
        super().__init__(**kwargs)

        self._input_shape = input_shape
        self._layers = layers
        self._dropout = dropout
        self._activation = activation

        init = Input(input_shape)
        # mlp_modules = []
        # for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
