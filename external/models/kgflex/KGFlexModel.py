"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Antonio Ferrara, Alberto Carlo Maria Mancino'
__email__ = 'vitowalter.anelli@poliba.it, antonio.ferrara@poliba.it, alberto.mancino@poliba.it'

import pickle

import numpy as np
from tqdm import tqdm


class KGFlexModel:
    def __init__(self,
                 data,
                 n_features,
                 learning_rate,
                 embedding_size,
                 user_features,
                 item_features,
                 feature_key_mapping):

        self._data = data
        self._learning_rate = learning_rate
        self._n_users = data.num_users
        self._n_items = data.num_items
        self._users = data.private_users.keys()

        self._n_features = n_features
        self._embedding_size = embedding_size

        # GLOBAL
        # Global features embeddings and bias
        self.Gf = np.random.randn(n_features, embedding_size) / 10
        self.Gb = np.random.randn(n_features) / 10

        # PERSONAL FEATURES
        # personal feature embeddings
        self.P_sp = {u: np.random.randn(len(user_features[u]), self._embedding_size) / 10
                     for u in self._users}
        # personal feature weights
        self.K_sp = {u: np.array([ig for ig in fs.values()]) for u, fs in user_features.items()}

        # USER FEATURE MAPPING
        # users features mask
        self.Mu = np.zeros(shape=(self._n_users, n_features)).astype(bool)
        for u, f in user_features.items():
            self.Mu[u][list(map(feature_key_mapping.get, f))] = True
        # item features mask
        self.Mi = np.zeros(shape=(self._n_items, n_features)).astype(np.bool)
        for i, row in enumerate(self.Mi):
            self.Mi[i][list(map(feature_key_mapping.get, item_features[i]))] = True

        # mapping global features indexing into user personal feature indexing
        u_ft_idx = {u: dict(zip(map(feature_key_mapping.get, uf_), range(len(uf_)))) for u, uf_ in
                    user_features.items()}

        # user item features
        self._n_features_range = np.array(range(self._n_features))

        self.user_item_features = {u: {i: self._n_features_range[np.multiply(mu, mi)] for i, mi in enumerate(self.Mi)}
                                   for u, mu in tqdm(enumerate(self.Mu), desc='user-item features', total=len(self.Mu))}

        self.user_item_feature_idxs = {u: {i: list(map(u_ft_idx[u].get, f)) for i, f in its.items()} for u, its in
                                       tqdm(self.user_item_features.items(), desc='user-item features indexed',
                                            total=len(self.user_item_features))}

    def __call__(self, *inputs):
        user, item = inputs
        user, item = np.array(user), np.array(item)
        return np.array(
            [np.sum((np.sum(
                np.multiply(self.P_sp[u][self.user_item_feature_idxs[u][i]], self.Gf[self.user_item_features[u][i]]),
                axis=1) + self.Gb[self.user_item_features[u][i]]) * self.K_sp[u][self.user_item_feature_idxs[u][i]])
             for u, i in zip(user, item)])

    def train_step(self, batch):

        user, pos, neg = batch
        user = user[:, 0]
        pos = pos[:, 0]
        neg = neg[:, 0]
        x_p = self(user, pos)
        x_n = self(user, neg)
        x_pn = np.subtract(x_p, x_n)
        e = np.exp(-x_pn)
        loss = np.sum(1 + e)
        d_loss = e / (1 + e)

        for us_, d_loss_, p, n in zip(user, d_loss, pos, neg):
            f_p = self.user_item_features[us_][p]
            f_n = self.user_item_features[us_][n]

            if len(f_n) > 0:

                f_p_sp = self.user_item_feature_idxs[us_][p]
                f_n_sp = self.user_item_feature_idxs[us_][n]

                p_term = d_loss_ * self.K_sp[us_][f_p_sp] * self._learning_rate
                n_term = -d_loss_ * self.K_sp[us_][f_n_sp] * self._learning_rate

                # updates
                self.P_sp[us_][f_p_sp] += self.Gf[f_p] * p_term[:, np.newaxis]
                self.Gf[f_p] += self.P_sp[us_][f_p_sp] * p_term[:, np.newaxis]
                self.Gb[f_p] += p_term

                self.P_sp[us_][f_n_sp] += self.Gf[f_n] * n_term[:, np.newaxis]
                self.Gf[f_n] += self.P_sp[us_][f_n_sp] * n_term[:, np.newaxis]
                self.Gb[f_n] += n_term
            else:
                pass

        return loss

    def predict(self, user):

        eval_items = list(range(self._n_items))

        p = self.P_sp[user]
        gf = self.Gf[self.Mu[user]]
        gb = self.Gb[self.Mu[user]]
        k = self.K_sp[user]

        f_interactions = (np.sum(np.multiply(p, gf), axis=1) + gb) * k

        return np.array([np.sum(f_interactions[self.user_item_feature_idxs[user][i]]) for i in eval_items])

    def get_user_recs(self, u, mask, k):
        user_id = self._data.public_users.get(u)
        user_recs = self.predict(user_id)
        user_recs_mask = mask[user_id]
        user_recs[~user_recs_mask] = -np.inf
        indices, values = zip(*[(self._data.private_items.get(u_list[0]), u_list[1])
                                for u_list in enumerate(user_recs)])
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
            '_user_feature_embeddings': self.P_sp,
            '_user_feature_weights': self.K_sp,
            '_item_feature_mask': self.Mi}
        return saving_dict

    def set_model_state(self, saving_dict):
        self.Gf = saving_dict['_global_features']
        self.Gb = saving_dict['_global_bias']
        self.Mu = saving_dict['_user_feature_mask']
        self.Mi = saving_dict['_item_feature_mask']
        self.P_sp = saving_dict['_user_feature_embeddings']
        self.K_sp = saving_dict['_user_feature_weights']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
