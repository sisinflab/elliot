import os
import random
from abc import ABC

import numpy as np
import tensorflow as tf

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class FeatureExtractor(tf.keras.Model, ABC):
    def __init__(self, k):
        super(FeatureExtractor, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(11, 11), strides=(4, 4), padding='same')
        self.relu1 = tf.keras.layers.ReLU()
        self.max1 = tf.keras.layers.MaxPool2D(padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same')
        self.relu2 = tf.keras.layers.ReLU()
        self.max2 = tf.keras.layers.MaxPool2D(padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.relu3 = tf.keras.layers.ReLU()
        self.conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.relu4 = tf.keras.layers.ReLU()
        self.conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.relu5 = tf.keras.layers.ReLU()
        self.max5 = tf.keras.layers.MaxPool2D(padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.f6 = tf.keras.layers.Dense(units=4096)
        self.relu6 = tf.keras.layers.ReLU()
        self.dropout6 = tf.keras.layers.Dropout(rate=0.5)
        self.f7 = tf.keras.layers.Dense(units=4096)
        self.relu7 = tf.keras.layers.ReLU()
        self.dropout7 = tf.keras.layers.Dropout(rate=0.5)
        self.f8 = tf.keras.layers.Dense(units=k)
        self.build((None, 224, 224, 3))

    def call(self, inputs, training=None, mask=None):
        conv1 = self.conv1(inputs)
        conv1 = self.relu1(conv1)
        conv1 = self.max1(conv1)

        conv2 = self.conv2(conv1)
        conv2 = self.relu2(conv2)
        conv2 = self.max2(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.relu3(conv3)

        conv4 = self.conv4(conv3)
        conv4 = self.relu4(conv4)

        conv5 = self.conv5(conv4)
        conv5 = self.relu5(conv5)
        conv5 = self.max5(conv5)

        fc1 = self.flatten(conv5)
        fc1 = self.f6(fc1)
        fc1 = self.relu6(fc1)
        fc1 = self.dropout6(inputs=fc1, training=training)

        fc2 = self.f7(fc1)
        fc2 = self.relu7(fc2)
        fc2 = self.dropout7(inputs=fc2, training=training)

        fc3 = self.f8(fc2)

        return fc3
