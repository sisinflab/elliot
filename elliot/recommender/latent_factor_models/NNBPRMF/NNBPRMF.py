"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import numpy as np
from utils import logging
from tqdm import tqdm

from dataset.samplers import custom_sampler as cs
from evaluation.evaluator import Evaluator
from utils.write import store_recommendation

from recommender import BaseRecommenderModel
from recommender.latent_factor_models.NNBPRMF.NNBPRMF_model import NNBPRMF_model
from recommender.recommender_utils_mixin import RecMixin

np.random.seed(42)


class NNBPRMF(RecMixin, BaseRecommenderModel):

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
        super().__init__(data, config, params, *args, **kwargs)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random
        self._sample_negative_items_empirically = True
        self._num_iters = self._params.epochs

        self._ratings = self._data.train_dict
        self._sampler = cs.Sampler(self._data.i_train_dict)
        self._iteration = 0
        self.evaluator = Evaluator(self._data, self._params)
        self._params.name = self.name

        ######################################

        self._factors = self._params.embed_k
        self._learning_rate = self._params.lr
        self._l_w = self._params.l_w
        self._l_b = self._params.l_b

        self._model = NNBPRMF_model(self._params.embed_k,
                                    self._params.lr,
                                    self._params.l_w,
                                    self._params.l_b,
                                    self._num_users,
                                    self._num_items)

        self._saving_filepath = f'{self._config.path_output_rec_weight}best-weights-{self.name}'
        self.logger = logging.get_logger(self.__class__.__name__)

    @property
    def name(self):
        return "BPR" \
               + "_lr:" + str(self._params.lr) \
               + "-e:" + str(self._params.epochs) \
               + "-factors:" + str(self._params.embed_k) \
               + "-br:" + str(self._params.l_b) \
               + "-wr:" + str(self._params.l_w)

    def train(self):
        best_metric_value = 0
        for it in range(self._num_iters):
            self.restore_weights(it)
            loss = 0
            steps = 0
            with tqdm(total=int(self._num_users // self._batch_size), disable=not self._verbose) as t:
                for batch in zip(*self._sampler.step(self._num_users, self._batch_size)):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            if not (it + 1) % self._verbose:
                recs = self.get_recommendations(self._config.top_k)
                results, statistical_results = self.evaluator.eval(recs)
                self._results.append(results)
                self._statistical_results.append(statistical_results)
                print(f'Epoch {(it + 1)}/{self._num_iters} loss {loss:.3f}')

                if self._results[-1][self._validation_metric] > best_metric_value:
                    print("******************************************")
                    best_metric_value = self._results[-1][self._validation_metric]
                    if self._save_weights:
                        self._model.save_weights(self._saving_filepath)
                    if self._save_recs:
                        store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        for index, offset in enumerate(range(0, self._num_users, self._params.batch_size)):
            offset_stop = min(offset+self._params.batch_size, self._num_users)
            predictions = self._model.predict_batch(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, self.get_train_mask(offset, offset_stop), k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
        return predictions_top_k
