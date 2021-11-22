# import multiprocessing
import numpy as np
# import math
import time
from tqdm import tqdm
from scipy.sparse import csr_matrix


class Server:

    def __init__(self, lr, model):
        self.model = model
        self.lr = lr
        # self.predictor = Predictor()

    def train_model(self, clients):
        tmp_feature_vecs = np.zeros(self.model.feature_vecs.shape)
        tmp_feature_bias = np.zeros(self.model.feature_bias.shape)
        for c in tqdm(clients):
            client_update = c.train(self.lr, self.model)
            tmp_feature_vecs += client_update[0]
            tmp_feature_bias += client_update[1]
        self.model.feature_vecs += tmp_feature_vecs
        self.model.feature_bias += tmp_feature_bias

    def predict(self, clients, users_mapping_reverse, items_mapping_reverse, mask, max_k):
        predictions = dict()
        for i, c in enumerate(clients):
            print('\r==== PREDICTIONS: CLIENT {} OF {} ===>'.format(i + 1, len(clients)), end='')
            predictions[users_mapping_reverse[c.user_id]] = c.predict(self.model, items_mapping_reverse,
                                                                      mask[c.user_id], max_k)
        print('\r✓ PREDICTIONS: {} CLIENTS\' PREDICTED'.format(len(clients)))
        return predictions
