"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from ast import literal_eval as make_tuple

from tqdm import tqdm
from .pointwise_pos_neg_sampler import Sampler
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .DeepCoNNModel import DeepCoNNModel

import numpy as np


class DeepCoNN(RecMixin, BaseRecommenderModel):
    r"""
    Joint Deep Modeling of Users and Items Using Reviews for Recommendation

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3018661.3018665>`_

    Args:
        batch_eval: Batch size for evaluation
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        u_rev_cnn_kernel: Tuple with kernel size for each user review cnn layer
        u_rev_cnn_features: Tuple with number of feature maps for each user review cnn layer
        i_rev_cnn_kernel: Tuple with kernel size for each item review cnn layer
        i_rev_cnn_features: Tuple with number of feature maps for each item review cnn layer
        latent_size: Latent size for the final fully-connected layer
        dropout: Dropout rate for each mlp layer in the model

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        DeepCoNN:
          meta:
            save_recs: True
          batch_eval: 64
          lr: 0.0005
          epochs: 50
          batch_size: 512
          l_w: 0.1
          u_rev_cnn_kernel: (3,)
          u_rev_cnn_features: (64,)
          i_rev_cnn_kernel: (3,)
          i_rev_cnn_features: (64,)
          latent_size: 128
          dropout: 0.5
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        iu_dict = data.build_items_neighbour()

        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 64, int, None),
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_u_rev_cnn_kernel", "u_rev_cnn_kernel", "u_rev_cnn_kernel", "(3,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_u_rev_cnn_features", "u_rev_cnn_features", "u_rev_cnn_features", "(64,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_i_rev_cnn_kernel", "i_rev_cnn_kernel", "i_rev_cnn_kernel", "(3,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_i_rev_cnn_features", "i_rev_cnn_features", "i_rev_cnn_features", "(64,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_latent_size", "latent_size", "latent_size", 128, int, None),
            ("_dropout", "dropout", "dropout", 0.5, float, None),
            ("_loader", "loader", "loader", 'WordsTextualAttributes', str, None)
        ]
        self.autoset_params()

        self._interactions_textual = self._data.side_information.WordsTextualAttributes

        self._pad_index = self._interactions_textual.object.word_features.shape[0] - 1

        self._sampler = Sampler(self._data.i_train_dict,
                                self._data.public_users,
                                self._data.public_items,
                                self._interactions_textual.object.users_tokens,
                                self._interactions_textual.object.items_tokens)

        self._model = DeepCoNNModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            l_w=self._l_w,
            vocabulary_features=self._interactions_textual.object.word_features,
            user_review_cnn_kernel=self._u_rev_cnn_kernel,
            user_review_cnn_features=self._u_rev_cnn_features,
            item_review_cnn_kernel=self._i_rev_cnn_kernel,
            item_review_cnn_features=self._i_rev_cnn_features,
            latent_size=self._latent_size,
            dropout_rate=self._dropout,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "DeepCoNN" \
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
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for user in range(self._num_users):
            predictions = np.empty((1, self._num_items))
            start_index = 0
            with tqdm(total=int(self._num_items // self._batch_eval), disable=not self._verbose) as t:
                for batch in self._sampler.step_eval(user, self._batch_eval):
                    u, i, rating, u_review_tokens, i_review_tokens = batch
                    end_index = start_index + u.shape[0]
                    predictions[0, start_index:end_index] = self._model.predict(batch)
                    start_index += u.shape[0]
                    t.update()
            recs_val, recs_test = self.process_protocol(k, predictions, user, user + 1)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test
