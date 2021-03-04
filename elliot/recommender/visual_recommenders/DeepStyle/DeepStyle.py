"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.visual_recommenders.DeepStyle.DeepStyle_model import DeepStyle_model
from elliot.utils.write import store_recommendation

np.random.seed(0)
tf.random.set_seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DeepStyle(RecMixin, BaseRecommenderModel):
    r"""
    DeepStyle: Learning User Preferences for Visual Recommendation

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3077136.3080658>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        DeepStyle:
          meta:
            save_recs: True
          lr: 0.0005
          epochs: 50
          factors: 100
          batch_size: 128
          l_w: 0.000025
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        super().__init__(data, config, params, *args, **kwargs)
        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random

        self._params_list = [
            ("_factors", "factors", "factors", 100, None, None),
            ("_learning_rate", "lr", "lr", 0.0005, None, None),
            ("_l_w", "l_w", "l_w", 0.000025, None, None)
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._sampler = cs.Sampler(self._data.i_train_dict)

        item_indices = [self._data.item_mapping[self._data.private_items[item]] for item in range(self._num_items)]

        self._model = DeepStyle_model(self._factors,
                                      self._learning_rate,
                                      self._l_w,
                                      self._data.visual_features[item_indices],
                                      self._data.visual_features.shape[1],
                                      self._num_users,
                                      self._num_items)

    @property
    def name(self):
        return "DeepStyle" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        best_metric_value = 0
        for it in range(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            if not (it + 1) % self._validation_rate:
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                result_dict = self.evaluator.eval(recs)
                self._results.append(result_dict)

                print(f'Epoch {(it + 1)}/{self._epochs} loss {loss  / steps:.3f}')

                if self._results[-1][self._validation_k]["val_results"][self._validation_metric] > best_metric_value:
                    print("******************************************")
                    best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                    if self._save_weights:
                        self._model.save_weights(self._saving_filepath)
                    if self._save_recs:
                        store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        for index, offset in enumerate(range(0, self._num_users, self._params.batch_size)):
            offset_stop = min(offset+self._params.batch_size, self._num_users)
            predictions = self._model.predict(offset, offset_stop)
            mask = self.get_train_mask(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, mask, k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
        return predictions_top_k


