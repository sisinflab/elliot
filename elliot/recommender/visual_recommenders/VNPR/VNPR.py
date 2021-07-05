"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

from ast import literal_eval as make_tuple

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from elliot.recommender.visual_recommenders.VNPR import pairwise_pipeline_sampler_vnpr as ppsv
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.visual_recommenders.VNPR.VNPR_model import VNPRModel


class VNPR(RecMixin, BaseRecommenderModel):
    r"""
    Visual Neural Personalized Ranking for Image Recommendation

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3159652.3159728>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        mf_factors:: Number of latent factors for Matrix Factorization
        mlp_hidden_size: Tuple with number of units for each multi-layer perceptron layer
        prob_keep_dropout: Dropout rate for multi-layer perceptron
        batch_size: Batch size
        batch_eval: Batch for evaluation
        l_w: Regularization coefficient

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        VNPR:
          meta:
            save_recs: True
          lr: 0.001
          epochs: 50
          mf_factors: 10
          mlp_hidden_size: (32, 1)
          prob_keep_dropout: 0.2
          batch_size: 64
          batch_eval: 64
          l_w: 0.001
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_batch_eval", "batch_eval", "be", 512, int, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_l_w", "l_w", "l_w", 0.001, None, None),
            ("_l_v", "l_v", "l_v", 0.001, None, None),
            ("_mf_factors", "mf_factors", "mffactors", 10, None, None),
            ("_mlp_hidden_size", "mlp_hidden_size", "mlpunits", "(32,1)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_dropout", "dropout", "drop", 0.2, None, None),
            ("_loader", "loader", "load", "VisualAttributes", None, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._side = getattr(self._data.side_information, self._loader, None)

        item_indices = [self._side.item_mapping[self._data.private_items[item]] for item in range(self._num_items)]

        self._sampler = ppsv.Sampler(self._data.i_train_dict,
                                     item_indices,
                                     self._side.visual_pca_feature_folder_path,
                                     self._epochs)

        self._next_batch = self._sampler.pipeline(self._data.transactions, self._batch_size)

        self._model = VNPRModel(self._num_users,
                                self._num_items,
                                self._mf_factors,
                                self._l_w,
                                self._l_v,
                                self._mlp_hidden_size,
                                self._dropout,
                                self._learning_rate,
                                self._side.visual_pca_features_shape,
                                self._seed)

        # only for evaluation purposes
        self._next_eval_batch = self._sampler.pipeline_eval(self._batch_eval)

    @property
    def name(self):
        return "VNPR" \
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
                mf_item_embedding_1 = self._model.item_mf_embedding_1(item_rel)
                mf_item_embedding_2 = self._model.item_mf_embedding_2(item_rel)
                p = self._model.predict_item_batch(offset, offset_stop, mf_item_embedding_1, mf_item_embedding_2,
                                                   tf.Variable(feat))
                predictions[:(offset_stop - offset), item_rel] = p
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)

            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test
