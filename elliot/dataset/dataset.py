import numpy as np
import pandas as pd
import scipy.sparse as sp


class DataSet(object):
    """
    Load train and test dataset
    """

    def __init__(self, config, params, *args, **kwargs):
        """
        Constructor of DataSet
        :param path_train_data: relative path for train file
        :param path_test_data: relative path for test file
        """
        path_train_data = config.path_train_data
        path_test_data = config.path_test_data
        self.config = config
        self.column_names = ['userId', 'itemId', 'rating']
        self.train_dataframe = pd.read_csv(path_train_data, sep="\t", header=None, names=self.column_names)
        #ATTENZIONE USERS E NUMERI SONO CALCOLATI SUL TRAIN MENTRE PRIMA SU ENTRAMBI, VERIFICARE
        self.users = list(self.train_dataframe['userId'].unique())
        self.num_users = len(self.users)
        self.num_items = self.train_dataframe['itemId'].nunique()
        self.transactions = len(self.train_dataframe)

        self.train_dataframe_dict = self.build_dict(self.train_dataframe, self.users)
        self.train = self.build_sparse(self.train_dataframe_dict, self.num_users, self.num_items)

        self.test_dataframe = pd.read_csv(path_test_data, sep="\t", header=None, names=self.column_names)
        self.test = self.prepare_test(self.test_dataframe)

        print('{0} - Loaded'.format(path_train_data))
        self.params = params
        self.args = args
        self.kwargs = kwargs

    def build_dict(self, dataframe, users):
        ratings = {}
        for u in users:
            sel_ = dataframe[dataframe['userId'] == u]
            ratings[u] = dict(zip(sel_['itemId'], sel_['rating']))
        return ratings

    def build_sparse(self, dataframe_dict, num_users, num_items):
        train = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        for user, user_items in dataframe_dict.items():
            for item, rating in user_items.items():
                if rating > 0:
                    train[user, item] = 1.0
        return train

    #non capisco bene cosa faccia
    def load_train_file_as_list(self, filename):
        # Get number of users and items
        u_ = 0
        self.train_list, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    self.train_list.append(items)
                    items = []
                    u_ += 1
                index += 1
                items.append(i)
                line = f.readline()
        self.train_list.append(items)

    def prepare_test(self, dataframe):
        return dataframe[["userId","itemId"]].values

    def get_test(self):
        return self.build_dict(self.test_dataframe, self.users)
