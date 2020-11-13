"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import scipy.sparse as sp
import numpy as np


class DataModel(object):
    def __init__(self, dataframe, ratings, random):
        self.dataframe = dataframe
        self.ratings = ratings
        self.random = random

        self.users = list(self.ratings.keys())
        self.items = list({k for a in self.ratings.values() for k in a.keys()})
        self.private_users = {p: u for p, u in enumerate(self.users)}
        self.public_users = {v: k for k, v in self.private_users.items()}
        self.private_items = {p: i for p, i in enumerate(self.items)}
        self.public_items = {v: k for k, v in self.private_items.items()}
        self.transactions = sum(len(v) for v in self.ratings.values())

        self.sp_train = self.build_sparse()

    def build_sparse(self):

        rows, cols = self.dataframe['userId'], self.dataframe['itemId']
        data = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                             shape=(len(self.users), len(self.items)))
        return data
