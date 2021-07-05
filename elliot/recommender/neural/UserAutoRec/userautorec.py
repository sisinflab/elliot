"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from tqdm import tqdm

from elliot.dataset.samplers import sparse_sampler as sp
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.neural.UserAutoRec.userautorec_model import UserAutoRecModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation


class UserAutoRec(RecMixin, BaseRecommenderModel):
    r"""
    AutoRec: Autoencoders Meet Collaborative Filtering (User-based)

    For further details, please refer to the `paper <https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf>`_

    Args:
        hidden_neuron: List of units for each layer
        lr: Learning rate
        l_w: Regularization coefficient

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        UserAutoRec:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          hidden_neuron: 500
          lr: 0.0001
          l_w: 0.001
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        AutoRec: Autoencoders Meet Collaborative Filtering
        Link: https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf
        Args:
            data:
            config:
            params:
            *args:
            **kwargs:
        """

        self._params_list = [
            ("_lr", "lr", "lr", 0.0001, None, None),
            ("_hidden_neuron", "hidden_neuron", "hidden_neuron", 500, None, None),
            ("_l_w", "l_w", "l_w", 0.001, None, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._sampler = sp.Sampler(self._data.sp_i_train)

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._model = UserAutoRecModel(self._data, self._num_users, self._num_items, self._lr,
                                       self._hidden_neuron, self._l_w, self._seed)

    @property
    def name(self):
        return "UserAutoRec" \
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

            self.evaluate(it, loss.numpy()/(it + 1))