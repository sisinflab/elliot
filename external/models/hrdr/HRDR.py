"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from ast import literal_eval as make_tuple

from tqdm import tqdm
from .pointwise_pipeline_sampler_hrdr import Sampler
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .HRDRModel import HRDRModel

import numpy as np


class HRDR(RecMixin, BaseRecommenderModel):
    r"""
    Hybrid neural recommendation with joint deep representation learning of ratings and reviews

    For further details, please refer to the `paper <https://www.sciencedirect.com/science/article/pii/S0925231219313207>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        user_projection_rating: Tuple with number of units for each user projection rating layer
        item_projection_rating: Tuple with number of units for each item projection rating layer
        user_review_cnn: Tuple with number of feature maps for each user review cnn layer
        item_review_cnn: Tuple with number of feature maps for each item review cnn layer
        user_review_attention: Tuple with number of units for each user attention layer
        item_review_attention: Tuple with number of units for each item attention layer
        user_final_representation: Tuple with number of units for each user final representation layer
        item_final_representation: Tuple with number of units for each item final representation layer
        dropout: Dropout rate for each mlp layer in the model

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        HRDR:
          meta:
            save_recs: True
          lr: 0.0005
          epochs: 50
          batch_size: 512
          factors: 64
          l_w: 0.1
          user_projection_rating: (64,)
          item_projection_rating: (64,)
          user_review_cnn: (64,)
          item_review_cnn: (64,)
          user_review_attention: (64,)
          item_review_attention: (64,)
          user_final_representation: (64,)
          item_final_representation: (64,)
          dropout: 0.5
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        iu_dict = data.build_items_neighbour()

        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_user_projection_rating", "user_projection_rating", "user_projection_rating", "(64,)",
             lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_user_review_cnn", "user_review_cnn", "user_review_cnn", "(64,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_item_review_cnn", "item_review_cnn", "item_review_cnn", "(64,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_user_review_attention", "user_review_attention", "user_review_attention", "(64,)",
             lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_item_review_attention", "item_review_attention", "item_review_attention", "(64,)",
             lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_user_final_representation", "user_final_representation", "user_final_representation", "(64,)",
             lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_item_final_representation", "item_final_representation", "item_final_representation", "(64,)",
             lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_dropout", "dropout", "dropout", 0.5, float, None),
            ("_loader", "loader", "loader", 'InteractionsTextualAttributes', str, None)
        ]
        self.autoset_params()

        self._interactions_textual = self._data.side_information.InteractionsTextualAttributes

        self._sampler = Sampler(self._data.i_train_dict,
                                self._data.private_users,
                                self._data.private_items,
                                iu_dict,
                                self._interactions_textual.interactions_path,
                                self._interactions_textual.textual_feature_folder_path,
                                self._interactions_textual.interactions_features_shape,
                                self._epochs)

        self._next_batch = self._sampler.pipeline(self._num_users, 1)

        self._model = HRDRModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            l_w=self._l_w,
            user_projection_rating=self._user_projection_rating,
            item_projection_rating=self._item_projection_rating,
            user_review_cnn=self._user_review_cnn,
            item_review_cnn=self._item_review_cnn,
            user_review_attention=self._user_review_attention,
            item_review_attention=self._item_review_attention,
            user_final_representation=self._user_final_representation,
            item_final_representation=self._item_final_representation,
            dropout=self._dropout,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "HRDR" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        loss = 0
        steps = 0
        it = 0
        with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
            for batch in self._next_batch:
                steps += 1
                loss += self._model.train_step(batch)
                t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                t.update()

                if steps == self._data.transactions // self._batch_size:
                    t.reset()
                    self.evaluate(it, loss.numpy() / steps)
                    it += 1
                    steps = 0
                    loss = 0

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for u in self._next_eval_user:
            predictions = np.zeros((1, self._num_items))
            user, user_ratings, user_reviews = u
            for i in self._next_eval_item:
                item, item_ratings, item_reviews = i
                inputs = user, item, user_ratings, item_ratings, user_reviews, item_reviews
                predictions = self._model.predict(inputs)
            recs_val, recs_test = self.process_protocol(k, predictions, user, user + 1)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test
