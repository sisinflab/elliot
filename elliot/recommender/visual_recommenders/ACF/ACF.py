"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

import os
from ast import literal_eval as make_tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import elliot.dataset.samplers.custom_sparse_sampler as css
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.visual_recommenders.ACF.ACF_model import ACF_model
from elliot.utils.write import store_recommendation

np.random.seed(0)
tf.random.set_seed(0)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ACF(RecMixin, BaseRecommenderModel):
    r"""
    Attentive Collaborative Filtering: Multimedia Recommendation with Item- and Component-Level Attention

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3077136.3080797>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        layers_component: Tuple with number of units for each attentive layer (component-level)
        layers_item: Tuple with number of units for each attentive layer (item-level)

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        ACF:
          meta:
            save_recs: True
          lr: 0.0005
          epochs: 50
          factors: 100
          batch_size: 128
          l_w: 0.000025
          layers_component: (64, 1)
          layers_item: (64, 1)
    """
    def __init__(self, data, config, params, *args, **kwargs):
        super().__init__(data, config, params, *args, **kwargs)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random

        self._layers_component = self._params.layers_component
        self._layers_item = self._params.layers_item

        self._params_list = [
            ("_factors", "factors", "factors", 100, None, None),
            ("_learning_rate", "lr", "lr", 0.0005, None, None),
            ("_l_w", "l_w", "l_w", 0.000025, None, None),
            ("_layers_component", "layers_component", "layers_component", "(64,1)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_layers_item", "layers_item", "layers_item", "(64,1)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-"))
        ]

        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._sampler = css.Sampler(self._data.i_train_dict, self._data.sp_i_train)

        item_indices = [self._data.item_mapping[self._data.private_items[item]] for item in range(self._num_items)]

        self._model = ACF_model(self._factors,
                                self._layers_component,
                                self._layers_item,
                                self._learning_rate,
                                self._l_w,
                                self._data.visual_features[item_indices],
                                self._data.sp_i_train.toarray(),
                                self._num_users,
                                self._num_items)


    @property
    def name(self):
        return "ACF" \
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
            with tqdm(total=int(self._num_users // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._num_users, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy()/steps:.5f}'})
                    t.update()

            if not (it + 1) % self._validation_rate:
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                result_dict = self.evaluator.eval(recs)
                self._results.append(result_dict)

                print(f'Epoch {(it + 1)}/{self._epochs} loss {loss/steps:.5f}')

                if self._results[-1][self._validation_k]["val_results"][self._validation_metric] > best_metric_value:
                    print("******************************************")
                    best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                    if self._save_weights:
                        self._model.save_weights(self._saving_filepath)
                    if self._save_recs:
                        store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

