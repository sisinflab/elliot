"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from tqdm import tqdm
import numpy as np
import gc

from elliot.dataset.samplers import sparse_sampler as ss
from elliot.evaluation.evaluator import Evaluator
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from .kavae_model import KnowledgeAwareVariationalAutoEncoder
from elliot.utils.write import store_recommendation
from elliot.recommender.knowledge_aware.kaHFM_batch.tfidf_utils import TFIDF
from elliot.recommender.recommender_utils_mixin import RecMixin


class KaVAE(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a BPR-MF instance.
        (see https://arxiv.org/pdf/1205.2618 for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w, l_b]: regularization,
                                      lr: learning rate}
        """

        # autoset params
        self._params_list = [
            ("_alpha", "alpha", "alpha", 1, float, None),
            ("_intermediate_dim", "intermediate_dim", "intermediate_dim", 600, int, None),
            ("_latent_dim", "latent_dim", "latent_dim", 200, int, None),
            ("_lambda", "reg_lambda", "reg_lambda", 0.01, None, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_dropout_rate", "dropout_pkeep", "dropout_pkeep", 1, None, None),
            ("_loader", "loader", "load", "KAHFMLoader", None, None),
        ]
        self.autoset_params()
        self._ratings = self._data.train_dict
        self._side = getattr(self._data.side_information, self._loader, None)
        self._sampler = ss.Sampler(self._data.sp_i_train)
        self._tfidf_obj = TFIDF(self._side.feature_map)
        self._tfidf = self._tfidf_obj.tfidf()
        self._user_profiles = self._tfidf_obj.get_profiles(self._ratings)

        self._user_factors = \
            np.zeros(shape=(len(self._data.users), len(self._side.features)))
        self._item_factors = \
            np.zeros(shape=(len(self._data.items), len(self._side.features)))

        for i, f_dict in self._tfidf.items():
            if i in self._data.items:
                for f, v in f_dict.items():
                    self._item_factors[self._data.public_items[i]][self._side.public_features[f]] = v

        for u, f_dict in self._user_profiles.items():
            for f, v in f_dict.items():
                self._user_factors[self._data.public_users[u]][self._side.public_features[f]] = v

        # self._iteration = 0
        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        # self._transactions_per_epoch = self._data.transactions
        self._dropout_rate = 1. - self._dropout_rate

        self._model = KnowledgeAwareVariationalAutoEncoder(self._num_items,
                                                           self._item_factors,
                                                           self._intermediate_dim,
                                                           self._latent_dim,
                                                           self._learning_rate,
                                                           self._dropout_rate,
                                                           self._lambda)

        # the total number of gradient updates for annealing
        self._total_anneal_steps = 200000
        # largest annealing parameter
        self._anneal_cap = 0.2

    @property
    def name(self):
        return "KaVAE" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        self._update_count = 0
        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._num_users // self._batch_size), disable=not self._verbose) as t:
                # for batch in zip(*self._sampler.step(self._data.transactions, self._batch_size)):
                for batch in self._sampler.step(self._num_users, self._batch_size):
                    steps += 1

                    if self._total_anneal_steps > 0:
                        anneal = min(self._anneal_cap, 1. * self._update_count / self._total_anneal_steps)
                    else:
                        anneal = self._anneal_cap

                    loss += self._model.train_step(batch, anneal)
                    gc.collect()
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()
                    self._update_count += 1

            self.evaluate(it, loss.numpy()/(it + 1))

