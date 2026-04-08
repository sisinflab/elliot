"""
Module description:

"""


import numpy as np
import torch
from tqdm import tqdm
from scipy import sparse as sp
from sklearn.utils.extmath import randomized_svd

from elliot.recommender.base_recommender import TraditionalRecommender
from elliot.utils.registry import model_registry


@model_registry.register()
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

    def __init__(self, params, interactions, seed, *args, **kwargs):
        super().__init__(params, interactions, seed, *args, **kwargs)

        self.user_vec, self.item_vec = None, None
        self.params_to_save = ['user_vec', 'item_vec']

    def initialize(self):
        t = tqdm()
        t.set_description("Computing")

        U, sigma, Vt = randomized_svd(self._interactions.sparse,
                                      n_components=self.factors,
                                      random_state=self._seed)
        s_Vt = sp.diags(sigma) * Vt

        t.set_description("Done")

        self.user_vec = U
        self.item_vec = s_Vt.T

    def predict(self, user_indices, item_indices=None):
        # Select only the embeddings in the current batch
        user_embeddings = self.user_vec[user_indices.numpy()]

        # Compute predictions
        if item_indices is None:
            item_embeddings = self.item_vec
            einsum_string = "be,ie->bi"
        else:
            item_embeddings = self.item_vec[item_indices.clamp(min=0).numpy()]
            einsum_string = "be,bse->bs"

        predictions = np.einsum(
            einsum_string, user_embeddings, item_embeddings
        )

        predictions = torch.from_numpy(predictions)
        return predictions
