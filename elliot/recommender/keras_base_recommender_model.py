from abc import ABC

import tensorflow as tf


class RecommenderModel(tf.keras.Model, ABC):
    def __init__(self, data, params, *args, **kwargs):
        """
        This class represents a recommender model. You can load a pretrained model
        by specifying its checkpoint path and use it for training/testing purposes.

        Args:
            data: data loader object
            params: dictionary with all parameters
        """
        super().__init__(*args, **kwargs)
        self.data = data
        self.num_items = data.num_items
        self.num_users = data.num_users
        self.params = params
        self.epochs = self.params.epochs
        self.batch_size = self.params.batch_size
        self.verbose = self.params.verbose
        self.restore_epochs = self.params.restore_epochs
