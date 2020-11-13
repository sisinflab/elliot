"""
Created on April 4, 2020
Tensorflow 2.1.0 implementation of APR.
@author Anonymized
"""
import logging
import os

import numpy as np
import tensorflow as tf

# from elliot.evaluation.Evaluator_old import Evaluator
from recommender.keras_base_recommender_model import RecommenderModel

np.random.seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Random(RecommenderModel):

    def __init__(self, data, path_output_rec_result, path_output_rec_weight, args):
        """
        Create a Random recommender.
        :param data: data loader object
        :param path_output_rec_result: path to the directory rec. results
        :param path_output_rec_weight: path to the directory rec. model parameters
        :param args: parameters
        """
        super(Random, self).__init__(data, path_output_rec_result, path_output_rec_weight, args.rec)
        # self.evaluator = Evaluator(self, data, args.k)
        self.best = 0

    def get_full_inference(self):
        """
        Get Full Predictions useful for Full Store of Predictions
        :return: The matrix of predicted values.
        """

        # Create a random matrix of weights?
        np.random.random((self.num_users, self.num_items))
        return tf.convert_to_tensor(np.random.random((self.num_users, self.num_items)))

    def train(self):
        self.evaluator.store_recommendation()