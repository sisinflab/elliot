"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import torch
from tqdm import tqdm
from scipy import sparse as sp
from sklearn.utils.extmath import randomized_svd

from elliot.recommender.base_recommender import TraditionalRecommender


class PureSVD(TraditionalRecommender):
    """
    PureSVD

    For further details, please refer to the `paper <https://link.springer.com/chapter/10.1007/978-0-387-85820-3_5>`_

    Args:
        factors: Number of latent factors
        seed: Random seed

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        PureSVD:
          meta:
            save_recs: True
          factors: 10
          seed: 42
    """

    # Model hyperparameters
    factors: int = 10

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger)

        self.user_vec, self.item_vec = None, None
        self.params_to_save = ['user_vec', 'item_vec']

    def initialize(self):
        fake_iter = tqdm(desc="Computing")

        U, sigma, Vt = randomized_svd(self._data.sp_i_train,
                                      n_components=self.factors,
                                      random_state=self._seed)
        s_Vt = sp.diags(sigma) * Vt
        fake_iter.set_description("Done")

        self.user_vec = U
        self.item_vec = s_Vt.T

    def predict_full(self, user_indices):
        u_embeddings_batch = self.user_vec[user_indices.numpy()]
        i_embeddings_all = self.item_vec

        predictions =  u_embeddings_batch @ i_embeddings_all.T

        predictions = torch.from_numpy(predictions)
        return predictions

    def predict_sampled(self, user_indices, item_indices):
        u_embeddings_batch = self.user_vec[user_indices.numpy()]
        i_embeddings_candidate = self.item_vec[item_indices.clamp(min=0).numpy()]

        predictions = np.einsum(
            "bi,bji->bj", u_embeddings_batch, i_embeddings_candidate
        )

        predictions = torch.from_numpy(predictions)
        return predictions
