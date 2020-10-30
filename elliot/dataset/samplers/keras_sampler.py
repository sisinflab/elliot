import scipy.sparse as sp
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import pandas as pd
from scipy.sparse import dok_matrix
from time import time

np.random.seed(0)

_user_input = None
_item_input_pos = None
_batch_size = None
_index = None
_model = None
_train = None
_test = None
_num_items = None

def _get_train_batch(i):
    """
    Generation of a batch in multiprocessing
    :param i: index to control the batch generation
    :return:
    """
    user_batch, item_pos_batch, item_neg_batch = [], [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_pos_batch.append(_item_input_pos[_index[idx]])
        j = np.random.randint(_num_items)
        while j in _train[_user_input[_index[idx]]]:
            j = np.random.randint(_num_items)
        item_neg_batch.append(j)
    return np.array(user_batch)[:, None], np.array(item_pos_batch)[:, None], np.array(item_neg_batch)[:, None]


class Sampler:
    def sampling(self):
        _user_input, _item_input_pos = [], []
        for (u, i) in self.train.keys():
            # positive instance
            _user_input.append(u)
            _item_input_pos.append(i)
        return _user_input, _item_input_pos

    def shuffle(self, batch_size=512):
        """
        Shuffle dataset to create batch with batch size
        Variable are global to be faster!
        :param batch_size: default 512
        :return: set of all generated random batches
        """
        global _user_input
        global _item_input_pos
        global _batch_size
        global _index
        global _model
        global _train
        global _num_items

        _user_input, _item_input_pos = self._user_input, self._item_input_pos
        _batch_size = batch_size
        _index = list(range(len(_user_input)))
        _train = self.train_list
        _num_items = self.num_items

        np.random.shuffle(_index)
        _num_batches = len(_user_input) // _batch_size
        pool = Pool(cpu_count())
        res = pool.map(_get_train_batch, range(_num_batches))
        pool.close()
        pool.join()

        user_input = [r[0] for r in res]
        item_input_pos = [r[1] for r in res]
        item_input_neg = [r[2] for r in res]
        return user_input, item_input_pos, item_input_neg
