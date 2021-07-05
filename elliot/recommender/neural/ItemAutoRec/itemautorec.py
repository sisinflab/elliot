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
from elliot.recommender.neural.ItemAutoRec.itemautorec_model import ItemAutoRecModel
from elliot.recommender.recommender_utils_mixin import RecMixin


class ItemAutoRec(RecMixin, BaseRecommenderModel):
    r"""
    AutoRec: Autoencoders Meet Collaborative Filtering (Item-based)

    For further details, please refer to the `paper <https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf>`_

    Args:
        hidden_neuron: List of units for each layer
        lr: Learning rate
        l_w: Regularization coefficient

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        ItemAutoRec:
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
        self._data.sp_u_train = self._data.sp_i_train.transpose()  # transpose the Matrix
        self._sampler = sp.Sampler(self._data.sp_u_train)

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._model = ItemAutoRecModel(self._data, self._num_users, self._num_items, self._lr,
                                       self._hidden_neuron, self._l_w, self._seed)

    @property
    def name(self):
        return "ItemAutoRec" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._num_items // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._num_items, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss.numpy()/(it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for batch in self._sampler.step(self._num_items, self._num_items):
            predictions = self._model.get_recs(batch)
        predictions = np.transpose(np.array(predictions))  # We have to build the transpose since we query the model by items.
        recs_val, recs_test = self.process_protocol(k, predictions, 0, self._data.num_users)
        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    # def get_recommendations(self, k: int = 100):
    #     predictions_top_k = {}
    #     for batch in self._sampler.step(self._num_items, self._num_items):
    #         predictions = self._model.get_recs(batch)
    #     predictions = np.transpose(
    #         np.array(predictions))  # We have to build the transpose since we query the model by items.
    #     v, i = self._model.get_top_k(predictions, self.get_train_mask(0, self._data.num_users), k=k)
    #     items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
    #                           for u_list in list(zip(i.numpy(), v.numpy()))]
    #     predictions_top_k.update(dict(zip(map(self._data.private_users.get,
    #                                           range(self._data.num_users)), items_ratings_pair)))
    #     return predictions_top_k
