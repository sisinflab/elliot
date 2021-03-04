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

from elliot.dataset.samplers import pipeline_sampler as ps
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.visual_recommenders.DVBPR.DVBPR_model import DVBPR_model
from elliot.utils.write import store_recommendation

np.random.seed(0)
tf.random.set_seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class DVBPR(RecMixin, BaseRecommenderModel):
    r"""
    Visually-Aware Fashion Recommendation and Design with Generative Image Models

    For further details, please refer to the `paper <https://doi.org/10.1109/ICDM.2017.30>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
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
          lambda_1: 0.0001
          lambda_2: 1.0
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        super().__init__(data, config, params, *args, **kwargs)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random

        self._params_list = [
            ("_factors", "factors", "factors", 100, None, None),
            ("_learning_rate", "lr", "lr", 0.0001, None, None),
            ("_lambda_1", "lambda_1", "lambda_1", 0.0001, None, None),
            ("_lambda_2", "lambda_2", "lambda_2", 1.0, None, None)
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict
        item_indices = [self._data.item_mapping[self._data.private_items[item]] for item in range(self._num_items)]

        self._sampler = ps.Sampler(
            self._data.i_train_dict,
            item_indices,
            self._data.side_information_data.images_src_folder,
            self._data.output_image_size,
            self._epochs
        )
        self._next_batch = self._sampler.pipeline(self._data.transactions, self._batch_size)

        # only for evaluation purposes
        self._next_image_batch = self._sampler.pipeline_eval(self._batch_size)

        self._model = DVBPR_model(self._factors,
                                  self._learning_rate,
                                  self._lambda_1,
                                  self._lambda_2,
                                  self._num_users,
                                  self._num_items)

    @property
    def name(self):
        return "DVBPR" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        best_metric_value = 0
        loss = 0
        steps = 0
        it = 0

        with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
            for batch in self._next_batch:
                steps += 1
                loss += self._model.train_step(batch)
                t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                t.update()

                # epoch is over
                if steps == self._data.transactions // self._batch_size:
                    t.reset()
                    if not (it + 1) % self._validation_rate:
                        recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                        result_dict = self.evaluator.eval(recs)
                        self._results.append(result_dict)

                        self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss / steps:.3f}')

                        if self._results[-1][self._validation_k]["val_results"][self._validation_metric] > best_metric_value:
                            best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                            if self._save_weights:
                                self._model.save_weights(self._saving_filepath)
                            if self._save_recs:
                                store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")
                    it += 1
                    steps = 0
                    loss = 0

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}

        # first, calculate all image features according to current model weights
        steps = 0
        visual_features = np.empty((self._num_items, self._factors))
        for im_batch in self._next_image_batch:
            im_id, im = im_batch
            output = self._model.Cnn(im, training=False).numpy()
            visual_features[steps:steps + output.shape[0]] = output
            steps += output.shape[0]

        for index, offset in enumerate(range(0, self._num_users, self._params.batch_size)):
            offset_stop = min(offset + self._params.batch_size, self._num_users)
            predictions = self._model.predict_batch(offset, offset_stop, tf.Variable(visual_features, dtype=tf.float32))
            mask = self.get_train_mask(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, mask, k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
        return predictions_top_k
