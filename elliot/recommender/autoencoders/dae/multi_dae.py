"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import random
from utils import logging

from dataset.samplers import sparse_sampler as sp
from evaluation.evaluator import Evaluator
from utils.folder import build_model_folder

from recommender import BaseRecommenderModel
from recommender.recommender_utils_mixin import RecMixin

from recommender.autoencoders.dae.multi_dae_utils import DenoisingAutoEncoder

np.random.seed(42)
random.seed(0)


class MultiDAE(RecMixin, BaseRecommenderModel):

    def __init__(self, data, config, params, *args, **kwargs):
        """
        """
        super().__init__(data, config, params, *args, **kwargs)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random
        self._random_p = random
        self._num_iters = self._params.epochs

        self._ratings = self._data.train_dict
        self._sampler = sp.Sampler(self._data.sp_i_train)
        self._iteration = 0
        self.evaluator = Evaluator(self._data, self._params)
        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params.name = self.name

        self._intermediate_dim = self._params.intermediate_dim
        self._latent_dim = self._params.latent_dim

        self._lambda = self._params.reg_lambda
        self._learning_rate = self._params.lr
        self._dropout_rate = 1. - self._params.dropout_pkeep

        self._model = DenoisingAutoEncoder(self._num_items,
                                           self._intermediate_dim,
                                           self._latent_dim,
                                           self._learning_rate,
                                           self._dropout_rate,
                                           self._lambda)

        build_model_folder(self._config.path_output_rec_weight, self.name)
        self._saving_filepath = f'{self._config.path_output_rec_weight}{self.name}/best-weights-{self.name}'
        self.logger = logging.get_logger(self.__class__.__name__)

    @property
    def name(self):
        return "MultiDAE" \
               + "_lr:" + str(self._params.lr) \
               + "-e:" + str(self._params.epochs) \
               + "-idim:" + str(self._params.intermediate_dim) \
               + "-ldim:" + str(self._params.latent_dim) \
               + "-bs:" + str(self._params.batch_size) \
               + "-dpk:" + str(self._params.dropout_pkeep) \
               + "-lmb:" + str(self._params.reg_lambda)

