"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle

import numpy as np

from elliot.utils import sparse


class MFModel(object):
    def __init__(self, F,
                 data,
                 lr,
                 reg,
                 random_seed,
                 *args):
        np.random.seed(random_seed)
        self._factors = F
        self._users = data.users
        self._items = data.items
        self._private_users = data.private_users
        self._public_users = data.public_users
        self._private_items = data.private_items
        self._public_items = data.public_items
        self._lr = lr
        self._reg = reg
        self.initialize(*args)

    def initialize(self, loc: float = 0, scale: float = 0.1):
        """
        This function initialize the data model
        :param loc:
        :param scale:
        :return:
        """

        self._global_bias = 0

        "same parameters as np.randn"
        self._user_bias = np.zeros(len(self._users))
        self._item_bias = np.zeros(len(self._items))
        self._user_factors = \
            np.random.normal(loc=loc, scale=scale, size=(len(self._users), self._factors))
        self._item_factors = \
            np.random.normal(loc=loc, scale=scale, size=(len(self._items), self._factors))
        #self._user_factors = sparse.create_sparse_matrix(user_factors_dense, float)
        #self._item_factors = sparse.create_sparse_matrix(item_factors_dense, float)

    @property
    def name(self):
        return "MF2020"

    def indexed_predict(self, user, item):
        return self._global_bias + self._user_bias[user] + self._item_bias[item] \
               + self._user_factors[user] @ self._item_factors[item]

    def get_top_k(self, preds, mask, k):
        #u_index = np.asarray(pr_batch)
        users_recs = self.apply_mask(preds, mask)
        index_ordered = np.argpartition(users_recs, -k, axis=1)[:, -k:]
        value_ordered = np.take_along_axis(users_recs, index_ordered, axis=1)
        local_top_k = np.take_along_axis(index_ordered, value_ordered.argsort(axis=1)[:, ::-1], axis=1)
        value_sorted = np.take_along_axis(users_recs, local_top_k, axis=1)
        return local_top_k, value_sorted

    """def get_user_predictions(self, user_id, mask, top_k=10):
        user_id = self._public_users.get(user_id)
        # b = self._train[user_id].dot(W_sparse)
        # b = self._global_bias + self._user_bias[user_id] + self._item_bias \
        #        + self._user_factors[user_id] @ self._item_factors.T
        b = self._preds[user_id]
        a = mask[user_id]
        b[~a] = -np.inf
        indices, values = zip(*[(self._private_items.get(u_list[0]), u_list[1])
                              for u_list in enumerate(b.data)])

        indices = np.array(indices)
        values = np.array(values)
        local_k = min(top_k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]"""

    def train_step(self, batch, **kwargs):
        sum_of_loss = 0
        lr = self._lr
        reg = self._reg
        for user, item, rating in batch:
            gb_ = self._global_bias
            uf_ = self._user_factors[user]
            if_ = self._item_factors[item]
            ub_ = self._user_bias[user]
            ib_ = self._item_bias[item]

            prediction = gb_ + ub_ + ib_ + np.dot(uf_, if_)
            #prediction = gb_ + ub_ + ib_ + uf_ @ if_

            if prediction > 0:
                one_plus_exp_minus_pred = 1.0 + np.exp(-prediction)
                sigmoid = 1.0 / one_plus_exp_minus_pred
                this_loss = (np.log(one_plus_exp_minus_pred) +
                             (1.0 - rating) * prediction)
            else:
                exp_pred = np.exp(prediction)
                sigmoid = exp_pred / (1.0 + exp_pred)
                this_loss = -rating * prediction + np.log(1.0 + exp_pred)

            grad = rating - sigmoid

            new_uf = uf_ + lr * (grad * if_ - reg * uf_)
            new_if = if_ + lr * (grad * uf_ - reg * if_)

            self._user_factors[user, :] = new_uf
            self._item_factors[item, :] = new_if
            self._user_bias[user] += lr * (grad - reg * ub_)
            self._item_bias[item] += lr * (grad - reg * ib_)
            self._global_bias += lr * (grad - reg * gb_)
            sum_of_loss += this_loss

        return sum_of_loss

    def predict(self, start, stop):
        return (np.expand_dims(self._user_bias[start:stop], axis=1) +
                (self._global_bias + self._item_bias + self._user_factors[start:stop] @ self._item_factors.T))

    """def update_factors(self, user: int, item: int, rating: float):
        uf_ = self._user_factors[user]
        if_ = self._item_factors[item]
        ub_ = self._user_bias[user]
        ib_ = self._item_bias[item]
        gb_ = self._global_bias
        lr = self._lr
        reg = self._reg


        prediction = gb_ + ub_ + ib_ + np.dot(uf_,if_)
        # prediction = gb_ + ub_ + ib_ + uf_ @ if_

        if prediction > 0:
            one_plus_exp_minus_pred = 1.0 + np.exp(-prediction)
            sigmoid = 1.0 / one_plus_exp_minus_pred
            this_loss = (np.log(one_plus_exp_minus_pred) +
                         (1.0 - rating) * prediction)
        else:
            exp_pred = np.exp(prediction)
            sigmoid = exp_pred / (1.0 + exp_pred)
            this_loss = -rating * prediction + np.log(1.0 + exp_pred)

        grad = rating - sigmoid

        self._user_factors[user] += lr * (grad * if_ - reg * uf_)
        self._item_factors[item] += lr * (grad * uf_ - reg * if_)
        self._user_bias[user] += lr * (grad - reg * ub_)
        self._item_bias[item] += lr * (grad - reg * ib_)
        self._global_bias += lr * (grad - reg * gb_)

        return this_loss"""

    def get_all_topks(self, mask, k, user_map, item_map):
        masking = np.where(mask, self._preds, -np.inf)
        partial_index = np.argpartition(masking, -k, axis=1)[:, -k:]
        masking_partition = np.take_along_axis(masking, partial_index, axis=1)
        masking_partition_index = masking_partition.argsort(axis=1)[:, ::-1]
        partial_index = np.take_along_axis(partial_index, masking_partition_index, axis=1)
        masking_partition = np.take_along_axis(masking_partition, masking_partition_index, axis=1)
        predictions_top_k = {
            user_map[u]: list(map(lambda x: (item_map.get(x[0]), x[1]), zip(*map(lambda x: x, top)))) for
            u, top in enumerate(zip(*(partial_index.tolist(), masking_partition.tolist())))}
        return predictions_top_k

    def get_model_state(self):
        saving_dict = {}
        saving_dict['_global_bias'] = self._global_bias
        saving_dict['_user_bias'] = self._user_bias
        saving_dict['_item_bias'] = self._item_bias
        saving_dict['_user_factors'] = self._user_factors
        saving_dict['_item_factors'] = self._item_factors
        return saving_dict

    def set_model_state(self, saving_dict):
        self._global_bias = saving_dict['_global_bias']
        self._user_bias = saving_dict['_user_bias']
        self._item_bias = saving_dict['_item_bias']
        self._user_factors = saving_dict['_user_factors']
        self._item_factors = saving_dict['_item_factors']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)