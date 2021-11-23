import numpy as np


class ServerModel:

    def __init__(self, features_mapping, embedding, random_seed=42):
        np.random.seed(random_seed)
        self.feature_vecs = np.random.randn(len(features_mapping), embedding) / 10
        self.feature_bias = np.random.randn(len(features_mapping)) / 10