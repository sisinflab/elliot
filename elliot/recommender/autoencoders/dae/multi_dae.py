"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import random

import numpy as np
from tqdm import tqdm

from elliot.dataset.samplers import sparse_sampler as sp
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.autoencoders.dae.multi_dae_model import DenoisingAutoEncoder
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation
from elliot.recommender.base_recommender_model import init_charger


class MultiDAE(RecMixin, BaseRecommenderModel):
    r"""
    Collaborative denoising autoencoder

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3178876.3186150>`_

    Args:
        intermediate_dim: Number of intermediate dimension
        latent_dim: Number of latent factors
        reg_lambda: Regularization coefficient
        lr: Learning rate
        dropout_pkeep: Dropout probaility

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        MultiDAE:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          intermediate_dim: 600
          latent_dim: 200
          reg_lambda: 0.01
          lr: 0.001
          dropout_pkeep: 1
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        """

        self._ratings = self._data.train_dict
        self._sampler = sp.Sampler(self._data.sp_i_train)
        self._iteration = 0

        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_intermediate_dim", "intermediate_dim", "intermediate_dim", 600, None, None),
            ("_latent_dim", "latent_dim", "latent_dim", 200, None, None),
            ("_lambda", "reg_lambda", "reg_lambda", 0.01, None, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_dropout_rate", "dropout_pkeep", "dropout_pkeep", 1, None, None)
        ]
        self.autoset_params()

        self._dropout_rate = 1. - self._dropout_rate

        self._model = DenoisingAutoEncoder(self._num_items,
                                           self._intermediate_dim,
                                           self._latent_dim,
                                           self._learning_rate,
                                           self._dropout_rate,
                                           self._lambda,
                                           self._seed)

    @property
    def name(self):
        return "MultiDAE" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._num_users // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._num_users, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy()/steps:.5f}'})
                    t.update()

            self.evaluate(it, loss/(it + 1))
