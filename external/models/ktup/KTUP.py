"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import math
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from . import rating_sampler as rs
from . import triple_sampler as ts
from .KTUPModel import jtup

np.random.seed(42)


class KTUP(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a BPR-MF instance.
        (see https://arxiv.org/pdf/1205.2618 for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w, l_b]: regularization,
                                      lr: learning rate}
        """
        self._random = np.random

        self._ratings = self._data.train_dict

        # autoset params
        self._params_list = [
            ("_l2_lambda", "l2_lambda", "l2",  1e-5, None, None),
            ("_embedding_size", "embedding_size", "es", 64, int, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_joint_ratio", "joint_ratio", "jr", 0.7, None, None),
            ("_L1", "L1_flag", "l1", True, None, None),
            ("_norm_lambda", "norm_lambda", "nl", 1, None, None),
            ("_kg_lambda", "kg_lambda", "kgl", 1, None, None),
            ("_use_st_gumbel", "use_st_gumbel", "gum", False, None, None),
            ("_loader", "loader", "load", "KGRec", None, None)
        ]
        self.autoset_params()
        self._step_to_switch = self._joint_ratio * 10
        self._side = getattr(self._data.side_information, self._loader, None)

        self._iteration = 0
        if self._batch_size < 1:
            self._batch_size = self._data.num_users

        triple_epoch_length = math.ceil(float(len(self._side.Xs)) / (1 - self._joint_ratio))
        rating_epoch_length = math.ceil(float(self._data.transactions) / self._joint_ratio)

        self._epoch_length = max(triple_epoch_length, rating_epoch_length)
        self._sampler = rs.Sampler(self._data.i_train_dict, self._epoch_length)
        self._triple_sampler = ts.Sampler(self._side.entity_to_idx, self._side.Xs, self._side.Xp, self._side.Xo, self._epoch_length)

        self._i_items_set = list(range(self._num_items))

        new_map = defaultdict(lambda: -1)
        new_map.update({self._data.public_items[i]: idx for i, idx in self._side.public_items_entitiesidx.items()})
        ######################################

        self._model = jtup(self._learning_rate, self._L1, self._l2_lambda, self._embedding_size,
                           self._data.num_users, self._data.num_items, len(self._side.entity_set),
                           len(self._side.predicate_set), new_map)

    @property
    def name(self):
        return "kTUP" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0

            if it % 10 < self._step_to_switch:
                with tqdm(total=int(self._epoch_length // self._batch_size), disable=not self._verbose) as t:
                    for batch in self._sampler.step(self._batch_size):
                        steps += 1
                        loss += self._model.train_step_rec(batch, is_rec=True)
                        t.set_postfix({'loss REC': f'{loss.numpy() / steps:.5f}'})
                        t.update()
            else:
                with tqdm(total=int(self._epoch_length // self._batch_size), disable=not self._verbose) as t:
                    for batch in self._triple_sampler.step(self._batch_size):
                        steps += 1
                        loss += self._model.train_step_kg(batch, is_rec=False, kg_lambda=self._kg_lambda)
                        t.set_postfix({'loss KGC': f'{loss.numpy() / steps:.5f}'})
                        t.update()

            self.evaluate(it, loss/(it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.get_recs(
                (
                    np.repeat(np.array(list(range(offset, offset_stop)))[:, None], repeats=self._num_items, axis=1),
                    np.array([self._i_items_set for _ in range(offset, offset_stop)])
                )
            )
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)

            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test
