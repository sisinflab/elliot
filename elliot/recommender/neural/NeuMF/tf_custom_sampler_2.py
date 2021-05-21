"""
Module description:
"""
__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'
import tensorflow as tf
import numpy as np
import random


class Sampler(tf.data.Dataset):

    def _generator(num_samples: int):
        r_int = np.random.randint
        for _ in range(num_samples):
            yield (r_int(Sampler._NUM_USERS),)

    def mapped_one(user):
        # pos_items = []
        i_s = []
        for u in user:
            pos_items_ = Sampler._TRAIN.indices[Sampler._TRAIN.indices[:, 0] == u][:,1]
            i = np.random.randint(pos_items_.shape[0])
            # pos_items.append(pos_items_)
            i_s.append(i)
        ciao = list(zip(user, i_s))
        # print(len(user), len(i_s), len(pos_items))
        return ciao

    def mapped_two(user, i):
        out = []
        r_int = np.random.randint
        for p, u in enumerate(user):
            pos_items_ = Sampler._TRAIN.indices[Sampler._TRAIN.indices[:, 0] == u][:, 1]
            for _ in range(Sampler._M):
                j = r_int(Sampler._NUM_ITEMS)
                while j in pos_items_:
                    j = r_int(Sampler._NUM_ITEMS)
                out.append((u, i[p], j))
        return out

    def convert_sparse_matrix_to_sparse_tensor(X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def __new__(cls, sp_i_train=None, m=None, num_users=None, num_items=None, transactions=None, batch_size=512, random_seed=42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        cls._TRAIN = cls.convert_sparse_matrix_to_sparse_tensor(sp_i_train)
        cls._M = m
        cls._NUM_USERS = num_users
        cls._NUM_ITEMS = num_items

        data = tf.data.Dataset.from_generator(generator=cls._generator,
                                              output_shapes=((),),
                                              output_types=(np.int64,),
                                              args=(transactions * m,))
        data = data.batch(batch_size=batch_size)
        def call_map_one(u):
            return tf.py_function(cls.mapped_one, inp=(u,), Tout=(np.int64, np.int64))
        data = data.map(map_func=call_map_one, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        def call_map_two(u, i):
            print("ciao")
            return tf.py_function(cls.mapped_two, inp=(u, i), Tout=(np.int64, np.int64, np.int64))
        data = data.map(map_func=call_map_two, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # data._indexed_ratings = indexed_ratings
        # data._users = list(data._indexed_ratings.keys())
        # data._nusers = len(data._users)
        # data._items = list({k for a in data._indexed_ratings.values() for k in a.keys()})
        # data._nitems = len(data._items)
        # data._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        # data._lui_dict = {u: len(v) for u, v in data._ui_dict.items()}
        # data._m = m
        # data._pos_generator = cls._pos_generator
        # data._pos = self._pos_generator(data._ui_dict)
        return data