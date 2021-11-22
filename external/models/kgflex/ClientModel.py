import numpy as np
from scipy.sparse import csr_matrix


class ClientModel:
    def __init__(self, embedding, random_seed=42):
        np.random.seed(random_seed)
        self.embedding = embedding
        self.user_vecs = None
        self.user_weights = None

    def init_vec(self, user_features, model_features_mapping, index_mask):
        self.user_vecs = np.zeros((len(model_features_mapping), self.embedding))
        self.user_vecs[index_mask] = np.random.randn(sum(index_mask), self.embedding) / 10
        self.user_vecs = csr_matrix(self.user_vecs)
        self.user_weights = csr_matrix([user_features[f] if f in user_features else 0 for f in model_features_mapping])
