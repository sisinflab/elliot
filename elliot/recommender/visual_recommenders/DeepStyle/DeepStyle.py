"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from tqdm import tqdm

import numpy as np
import tensorflow as tf

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.visual_recommenders.DeepStyle.DeepStyle_model import DeepStyleModel
from elliot.recommender.visual_recommenders.DeepStyle import pairwise_pipeline_sampler_deepstyle as ppsd


class DeepStyle(RecMixin, BaseRecommenderModel):
    r"""
    DeepStyle: Learning User Preferences for Visual Recommendation

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3077136.3080658>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        batch_eval: Batch size for evaluation
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
          batch_eval: 512
          l_w: 0.000025
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_batch_eval", "batch_eval", "be", 512, int, None),
            ("_factors", "factors", "factors", 100, None, None),
            ("_learning_rate", "lr", "lr", 0.0005, None, None),
            ("_l_w", "l_w", "l_w", 0.000025, None, None),
            ("_loader", "loader", "load", "VisualAttributes", None, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._side = getattr(self._data.side_information, self._loader, None)

        item_indices = [self._side.item_mapping[self._data.private_items[item]] for item in range(self._num_items)]

        self._sampler = ppsd.Sampler(self._data.i_train_dict,
                                     item_indices,
                                     self._side.visual_feature_folder_path,
                                     self._epochs)

        self._next_batch = self._sampler.pipeline(self._data.transactions, self._batch_size)

        self._model = DeepStyleModel(self._factors,
                                     self._learning_rate,
                                     self._l_w,
                                     self._side.visual_features_shape,
                                     self._num_users,
                                     self._num_items,
                                     self._seed)

        # only for evaluation purposes
        self._next_eval_batch = self._sampler.pipeline_eval(self._batch_eval)

    @property
    def name(self):
        return "DeepStyle" \
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
        for index, offset in enumerate(range(0, self._num_users, self._batch_eval)):
            offset_stop = min(offset + self._batch_eval, self._num_users)
            predictions = np.empty((offset_stop - offset, self._num_items))
            for batch in self._next_eval_batch:
                item_rel, item_abs, feat = batch
                p = self._model.predict_item_batch(offset, offset_stop,
                                                   item_rel[0], item_rel[-1],
                                                   tf.Variable(feat))
                predictions[:(offset_stop - offset), item_rel] = p
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test
