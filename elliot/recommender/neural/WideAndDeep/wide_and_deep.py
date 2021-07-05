"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'
__paper__ = 'Wide & Deep Learning for Recommender Systems [https://dl.acm.org/doi/pdf/10.1145/2988450.2988454]'

from ast import literal_eval as make_tuple

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from elliot.dataset.samplers import pointwise_wide_and_deep_sampler as pwwds
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.neural.WideAndDeep.wide_and_deep_model import WideAndDeepModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation


def build_sparse_features(data):
    side_information_data = data.side_information_data

    sp_i_f = []
    for key_side_feature_type, value in vars(side_information_data).items():
        # print(key_side_feature_type)
        rows_cols = [(data.public_items[item], data.public_features[f]) for item, features in value.items() for
                     f in features]
        rows = [item for item, _ in rows_cols]
        cols = [f for _, f in rows_cols]
        sp_i_f.append(sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                             shape=(data.num_items, len(data.features))))

    # We will add user features

    user_encoder = OneHotEncoder()
    user_encoder.fit(np.reshape(np.arange(data.num_users), newshape=(data.num_users, 1)))
    item_encoder = OneHotEncoder()
    item_encoder.fit(np.reshape(np.arange(data.num_items), newshape=(data.num_items, 1)))

    return sp_i_f, user_encoder, item_encoder


class WideAndDeep(RecMixin, BaseRecommenderModel):
    r"""
    Wide & Deep Learning for Recommender Systems

    (For now, available with knowledge-aware features)

    For further details, please refer to the `paper <https://arxiv.org/abs/1606.07792>`_

    Args:
        factors: Number of latent factors
        mlp_hidden_size: List of units for each layer
        lr: Learning rate
        l_w: Regularization coefficient
        l_b: Bias Regularization Coefficient
        dropout_prob: Dropout rate

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        WideAndDeep:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 50
          mlp_hidden_size: (32, 32, 1)
          lr: 0.001
          l_w: 0.005
          l_b: 0.0005
          dropout_prob: 0.0
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._data.sp_i_features, self._data.user_encoder, self._data.item_encoder = build_sparse_features(self._data)

        self._sampler = pwwds.Sampler(self._data)

        self._params_list = [
            ("_lr", "lr", "lr", 0.001, None, None),
            ("_factors", "factors", "factors", 50, None, None),
            ("_mlp_hidden_size", "mlp_hidden_size", "mlp_hidden_size", "(32, 32, 1)",
             lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_dropout_prob", "dropout_prob", "dropout_prob", 0, None, None),
            ("_l_w", "l_w", "l_w", 0.005, None, None),
            ("_l_b", "l_b", "l_b", 0.0005, None, None)
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._model = WideAndDeepModel(self._data, self._num_users, self._num_items, self._factors,
                                       self._mlp_hidden_size,
                                       self._dropout_prob, self._lr, self._l_w, self._l_b,
                                       self._seed
                                       )

    @property
    def name(self):
        return "WideAndDeep" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss.numpy()/(it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test