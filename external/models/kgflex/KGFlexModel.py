"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import pickle

import tensorflow as tf
import numpy as np

_RANDOM_SEED = 42


class KGFlexModel():
    def __init__(self,
                 data,
                 learning_rate,
                 n_features,
                 feature_key_mapping,
                 item_features_mapper,
                 embedding_size,
                 index_mask,
                 users_features,
                 random_seed=_RANDOM_SEED,
                 **kwargs):

        tf.random.set_seed(random_seed)

        self._data = data
        self._learning_rate = learning_rate
        self._n_users = data.num_users
        self._n_items = data.num_items
        self._users = data.private_users.keys()

        self._n_features = n_features
        self._embedding_size = embedding_size

        # Global features embeddings
        self.Gf = np.random.randn(n_features, embedding_size) / 10
        self.Gb = np.random.randn(n_features) / 10

        # Personal features embeddings
        # users features mask
        self.Mu = np.zeros(shape=(self._n_users, n_features))
        for k, m in index_mask.items():
            self.Mu[k] = m
        self.Mu = self.Mu.astype(bool)

        # users features mask along embedding axis
        self.Me = np.repeat(self.Mu[:, :, np.newaxis], embedding_size, axis=2)
        # initialize users features vectors
        self.P = np.zeros((self._n_users, n_features, embedding_size))
        self.P[self.Me] = np.random.randn(*self.P[self.Me].shape) / 10

        # users constant weights
        self.K = np.zeros(shape=(self._n_users, n_features))
        for u in self._users:
            for feature in users_features[u].keys():
                self.K[u][feature_key_mapping[feature]] = users_features[u][feature]

        # item features mask
        self.Mi = np.zeros(shape=(self._n_items, n_features))
        for item, features in item_features_mapper.items():
            for f in features:
                key = feature_key_mapping.get(f)
                if key:
                    self.Mi[item][key] = 1
        self.Mi = self.Mi.astype(bool)

    def __call__(self, *inputs):

        # return x_ui
        user, item = inputs
        user, item = np.array(user), np.array(item)
        batch_size = user.shape[0]

        p = self.P[user]
        k = self.K[user]
        m = np.multiply(self.Mu[user], self.Mi[item])
        indexes = np.array(range(self._n_features))
        feature_indexes = []
        for u in range(batch_size):
            feature_indexes.append(indexes[m[u]])

        x_ui = []

        for u in range(batch_size):
            f = feature_indexes[u]
            # user-common features-embedding
            p_uf = p[u, f, :]
            # common features-embedding
            g_f = self.Gf[f, :]
            # bias on common features
            b_f = self.Gb[f]
            # weights of common features
            kk = k[u][f]
            # user item interaction
            x_ui.append(np.sum(((np.sum(np.multiply(p_uf, g_f), axis=1) + b_f) * kk)))

        #np.array(np.sum(((np.sum(np.multiply(p[u, f, :], self.Gf[f, :]), axis=1) + self.Gb[f]) * k[u][f])) for u, f in [(u, feature_indexes[u]) for u in range(batch_size)])

        return np.array(x_ui)

    def train_step(self, batch):

        loss = 0.0
        # user, pos, neg = batch
        # for sample in zip(*batch):
        user, pos, neg = batch
        user = user[:, 0]
        pos = pos[:, 0]
        neg = neg[:, 0]
        x_p = self(user, pos)
        x_n = self(user, neg)
        x_pn = np.subtract(x_p, x_n)
        d_loss = (1 / (1 + np.exp(x_pn)))

        m_p = np.multiply(self.Mu[user], self.Mi[pos])
        m_n = np.multiply(self.Mu[user], self.Mi[neg])

        indexes_p = np.array(range(self._n_features))
        indexes_n = np.array(range(self._n_features))

        for us_, mask_pos, mask_neg, d_loss_ in zip(user, m_p, m_n, d_loss):
            f_p = indexes_p[mask_pos]
            f_n = indexes_n[mask_neg]

            # updates
            self.P[us_, f_p] += self._learning_rate * self.Gf[f_p] * self.K[us_][f_p][:, np.newaxis] * d_loss_
            self.Gf[f_p] += self._learning_rate * self.P[us_, f_p] * self.K[us_][f_p][:, np.newaxis] * d_loss_
            self.Gb[f_p] += self._learning_rate * self.K[us_][f_p] * d_loss_

            self.P[us_, f_n] += self._learning_rate * self.Gf[f_n] * -self.K[us_][f_n][:, np.newaxis] * d_loss_
            self.Gf[f_n] += self._learning_rate * self.P[us_, f_n] * -self.K[us_][f_n][:, np.newaxis] * d_loss_
            self.Gb[f_n] += self._learning_rate * -self.K[us_][f_n] * d_loss_

            loss += d_loss_
        return loss

    def predict(self, user):
        eval_user = [user] * self._n_items
        eval_items = list(range(self._n_items))
        results = self((eval_user, eval_items))
        return {user: list(zip(eval_items, results))}

    def get_user_recs(self, u, mask, k):
        user_id = self._data.public_users.get(u)
        eval_user = [user_id] * self._n_items
        eval_items = list(range(self._n_items))
        user_recs = self(eval_user, eval_items)
        user_recs_mask = mask[user_id]
        user_recs[~user_recs_mask] = -np.inf
        indices, values = zip(*[(self._data.private_items.get(u_list[0]), u_list[1])
                                for u_list in enumerate(user_recs)])
        # indices, values = zip(*predictions.items())
        indices = np.array(indices)
        values = np.array(values)
        local_k = min(k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]

    def get_config(self):
        raise NotImplementedError

    def get_model_state(self):
        saving_dict = {
            '_global_features': self.Gf,
            '_global_bias': self.Gb,
            '_user_feature_mask': self.Mu,
            'user_feature_mask_along_embedding': self.Me,
            '_user_feature_embeddings': self.P,
            '_user_feature_weights': self.K,
            '_item_feature_mask': self.Mi}
        return saving_dict

    def set_model_state(self, saving_dict):
        self.Gf = saving_dict['_global_features']
        self.Gb = saving_dict['_global_bias']
        self.Mu = saving_dict['_user_feature_mask']
        self.Me = saving_dict['user_feature_mask_along_embedding']
        self.P = saving_dict['_user_feature_embeddings']
        self.K = saving_dict['_user_feature_weights']
        self.Mi = saving_dict['_item_feature_mask']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
