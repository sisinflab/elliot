"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from tqdm import tqdm

from elliot.dataset.samplers import sparse_sampler as sp
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.autoencoders.vae.multi_vae_model import VariationalAutoEncoder
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class MultiVAE(RecMixin, BaseRecommenderModel):
    r"""
    Variational Autoencoders for Collaborative Filtering

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
        MultiVAE:
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


        ######################################

        self._params_list = [
            ("_intermediate_dim", "intermediate_dim", "intermediate_dim", 600, int, None),
            ("_latent_dim", "latent_dim", "latent_dim", 200, int, None),
            ("_lambda", "reg_lambda", "reg_lambda", 0.01, None, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_dropout_rate", "dropout_pkeep", "dropout_pkeep", 1, None, None),
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict
        self._sampler = sp.Sampler(self._data.sp_i_train)

        if self._batch_size < 1:
            self._batch_size = self._num_users

        self._dropout_rate = 1. - self._dropout_rate

        self._model = VariationalAutoEncoder(self._num_items,
                                             self._intermediate_dim,
                                             self._latent_dim,
                                             self._learning_rate,
                                             self._dropout_rate,
                                             self._lambda,
                                             self._seed)

        # the total number of gradient updates for annealing
        self._total_anneal_steps = 200000
        # largest annealing parameter
        self._anneal_cap = 0.2

    @property
    def name(self):
        return "MultiVAE" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        self._update_count = 0

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._num_users // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._num_users, self._batch_size):
                    steps += 1

                    if self._total_anneal_steps > 0:
                        anneal = min(self._anneal_cap, 1. * self._update_count / self._total_anneal_steps)
                    else:
                        anneal = self._anneal_cap

                    loss += self._model.train_step(batch, anneal).numpy()
                    t.set_postfix({'loss': f'{loss/steps:.5f}'})
                    t.update()
                    self._update_count += 1

            self.evaluate(it, loss/(it + 1))
