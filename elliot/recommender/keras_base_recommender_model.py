from abc import ABC, abstractmethod

import tensorflow as tf


class RecommenderModel(tf.keras.Model, ABC):
    def __init__(self, config, params, *args, **kwargs):
        """
        This class represents a recommender model. You can load a pretrained model
        by specifying its checkpoint path and use it for training/testing purposes.

        Args:
            data: data loader object
            params: dictionary with all parameters
        """
        super().__init__(*args, **kwargs)
        self.config = config
        self.params = params

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def get_recommendations(self, *args):
        pass

    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def get_results(self):
        pass
