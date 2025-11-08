"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle
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
    factors: int = 10

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger)

        self.user_vec, self.item_vec = None, None
        self.params_to_save = ['user_vec', 'item_vec']

    def predict(self, start, stop):
        return self.user_vec[start:stop].dot(self.item_vec)

    def initialize(self):
        fake_iter = tqdm(range(1), desc="Computing")

        for _ in fake_iter:
            U, sigma, Vt = randomized_svd(self._data.sp_i_train,
                                          n_components=self.factors,
                                          random_state=self._seed)
            s_Vt = sp.diags(sigma) * Vt
            fake_iter.set_description("Done")

        self.user_vec = U
        self.item_vec = s_Vt
