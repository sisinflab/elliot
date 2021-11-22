import numpy as np
from scipy.sparse import csr_matrix
import time


class Client:
    """ class Client: contains the client id, the relative training subset and Client model """

    def __init__(self, user_id, model, user_data, user_features, model_features_mapping, upc,
                 item_features_mask, random_seed=42):
        np.random.seed(random_seed)
        self.user_id = user_id

        self.pos_items = set(user_data)
        self.neg_items = list(set(range(item_features_mask.shape[0])) - self.pos_items)
        self.pos_items = list(self.pos_items)
        self.update_per_client = upc if upc else len(self.pos_items)

        self.model = model

        index_mask = [True if f in user_features else False for f in model_features_mapping]
        self.index_mask = csr_matrix(index_mask)
        self.user_item_features_mask = self.index_mask.multiply(item_features_mask)

        self.model.init_vec(user_features, model_features_mapping, index_mask)

    def predict(self, server_model, items_mapping_reverse, mask, max_k):
        result = (self.model.user_weights.multiply(self.user_item_features_mask) * (
                self.model.user_vecs.multiply(server_model.feature_vecs).sum(
                    axis=1) + csr_matrix(server_model.feature_bias).T)).A1

        result[~mask] = -np.inf

        unordered_top_k = np.argpartition(result, -max_k)[-max_k:]
        top_k = unordered_top_k[np.argsort(result[unordered_top_k])][::-1]
        top_k_score = result[top_k]
        prediction = [(items_mapping_reverse[top_k[i]], top_k_score[i]) for i in range(len(top_k))]

        return prediction

    def train(self, lr, server_model):
        positive_sampled = np.random.choice(self.pos_items, self.update_per_client)
        negative_sampled = np.random.choice(self.neg_items, self.update_per_client)

        weighted_features = (self.model.user_weights.multiply(
            self.model.user_vecs.multiply(server_model.feature_vecs).sum(
                axis=1).A1 + server_model.feature_bias).toarray()).reshape(-1)

        x_p = weighted_features * self.user_item_features_mask[positive_sampled].T
        x_n = weighted_features * self.user_item_features_mask[negative_sampled].T

        # remove items with no prediction (no common features)
        deleted = np.concatenate((np.where(x_p == 0)[0], np.where(x_n == 0)[0]))
        x_p = np.delete(x_p, deleted)
        x_n = np.delete(x_n, deleted)
        positive_sampled = np.delete(positive_sampled, deleted)
        negative_sampled = np.delete(negative_sampled, deleted)

        if len(positive_sampled) > 0:
            x_pn = np.subtract(x_p, x_n)
            d_loss = (1 / (1 + np.exp(x_pn)))

            uv_ = self.model.user_vecs.A
            fv_ = np.array(server_model.feature_vecs, order='K', copy=True)

            pos_masked_weights = self.user_item_features_mask[positive_sampled].multiply(self.model.user_weights)
            pos_d_loss_masked_weights = (d_loss * pos_masked_weights.A.T).T

            neg_masked_weights = self.user_item_features_mask[negative_sampled].multiply(self.model.user_weights)
            neg_d_loss_masked_weights = (- d_loss * neg_masked_weights.A.T).T

            d_loss_masked_weights = pos_d_loss_masked_weights + neg_d_loss_masked_weights

            self.model.user_vecs += csr_matrix(
                lr * (np.sum(d_loss_masked_weights, axis=0)[:, None] * fv_))
            feature_vecs_update = lr * (
                         np.sum(d_loss_masked_weights, axis=0)[:, None] * uv_)
            feature_bias_update = lr * (d_loss.dot(pos_masked_weights.A) - d_loss.dot(neg_masked_weights.A))

            return feature_vecs_update, feature_bias_update

        return 0, 0
