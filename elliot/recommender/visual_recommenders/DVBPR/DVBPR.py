"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from elliot.recommender.visual_recommenders.DVBPR import pairwise_pipeline_sampler_dvbpr as ppsd
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.visual_recommenders.DVBPR.DVBPR_model import DVBPRModel


class DVBPR(RecMixin, BaseRecommenderModel):
    r"""
    Visually-Aware Fashion Recommendation and Design with Generative Image Models

    For further details, please refer to the `paper <https://doi.org/10.1109/ICDM.2017.30>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        batch_eval: Batch for evaluation
        lambda_1: Regularization coefficient
        lambda_2: CNN regularization coefficient

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        DVBPR:
          meta:
            save_recs: True
          lr: 0.0001
          epochs: 50
          factors: 100
          batch_size: 128
          batch_eval: 128
          lambda_1: 0.0001
          lambda_2: 1.0
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_batch_eval", "batch_eval", "be", 512, int, None),
            ("_factors", "factors", "factors", 100, None, None),
            ("_learning_rate", "lr", "lr", 0.0001, None, None),
            ("_lambda_1", "lambda_1", "lambda_1", 0.0001, None, None),
            ("_lambda_2", "lambda_2", "lambda_2", 1.0, None, None),
            ("_loader", "loader", "load", "VisualAttributes", None, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._side = getattr(self._data.side_information, self._loader, None)

        self._item_indices = [self._side.item_mapping[self._data.private_items[item]] for item in range(self._num_items)]

        self._sampler = ppsd.Sampler(
            self._data.i_train_dict,
            self._item_indices,
            self._side.images_folder_path,
            self._side.image_size_tuple,
            self._epochs
        )
        self._next_batch = self._sampler.pipeline(self._data.transactions, self._batch_size)

        self._model = DVBPRModel(self._factors,
                                 self._learning_rate,
                                 self._lambda_1,
                                 self._lambda_2,
                                 self._num_users,
                                 self._num_items,
                                 self._seed)

    @property
    def name(self):
        return "DVBPR" \
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

        # first, calculate all image features according to current model weights
        features = np.zeros(shape=(len(self._item_indices), self._factors))
        for start_batch in range(0, len(self._item_indices), self._batch_eval):
            stop_batch = min(start_batch + self._batch_eval, len(self._item_indices))
            images = np.zeros(shape=(stop_batch - start_batch, *self._side.image_size_tuple, 3))
            for start_image in range(start_batch, stop_batch):
                _, image = self._sampler.read_image(self._item_indices[start_image])
                images[start_image % self._batch_eval] = image
            features[start_batch:stop_batch] = self._model.Cnn(images, training=False).numpy()

        for index, offset in enumerate(range(0, self._num_users, self._batch_eval)):
            offset_stop = min(offset + self._batch_eval, self._num_users)
            predictions = np.empty((offset_stop - offset, self._num_items))
            for item_index, item_offset in enumerate(range(0, self._num_items, self._batch_eval)):
                item_offset_stop = min(item_offset + self._batch_eval, self._num_items)
                p = self._model.predict_item_batch(offset, offset_stop,
                                                   tf.Variable(features[item_index * self._batch_eval:item_offset_stop],
                                                               dtype=tf.float32))
                predictions[:(offset_stop - offset), item_index * self._batch_eval:item_offset_stop] = p
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test
