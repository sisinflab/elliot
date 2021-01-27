"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import random
from tqdm import tqdm

from dataset.samplers import sparse_sampler as sp
from evaluation.evaluator import Evaluator
from recommender import BaseRecommenderModel
from utils.folder import build_model_folder
from utils.write import store_recommendation

from recommender.autoencoders.vae.multi_vae_utils import VariationalAutoEncoder
from recommender.recommender_utils_mixin import RecMixin

np.random.seed(42)
random.seed(0)


class MultiVAE(RecMixin, BaseRecommenderModel):

    def __init__(self, data, config, params, *args, **kwargs):
        """
        """
        super().__init__(data, config, params, *args, **kwargs)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random
        self._random_p = random

        self._ratings = self._data.train_dict
        self._sampler = sp.Sampler(self._data.sp_i_train)
        self._iteration = 0
        self.evaluator = Evaluator(self._data, self._params)
        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params.name = self.name

        self._intermediate_dim = self._params.intermediate_dim
        self._latent_dim = self._params.latent_dim

        self._lambda = self._params.reg_lambda
        self._learning_rate = self._params.lr
        self._dropout_rate = 1. - self._params.dropout_pkeep

        self._model = VariationalAutoEncoder(self._num_items,
                                           self._intermediate_dim,
                                           self._latent_dim,
                                           self._learning_rate,
                                           self._dropout_rate,
                                           self._lambda)

        # the total number of gradient updates for annealing
        self._total_anneal_steps = 200000
        # largest annealing parameter
        self._anneal_cap = 0.2

        build_model_folder(self._config.path_output_rec_weight, self.name)
        self._saving_filepath = f'{self._config.path_output_rec_weight}{self.name}/best-weights-{self.name}'

    @property
    def name(self):
        return "MultiVAE" \
               + "_lr:" + str(self._params.lr) \
               + "-e:" + str(self._params.epochs) \
               + "-idim:" + str(self._params.intermediate_dim) \
               + "-ldim:" + str(self._params.latent_dim) \
               + "-bs:" + str(self._params.batch_size) \
               + "-dpk:" + str(self._params.dropout_pkeep) \
               + "-lmb" + str(self._params.reg_lambda)

    def train(self):

        best_metric_value = 0
        self._update_count = 0

        for it in range(self._epochs):
            self.restore_weights(it)
            loss = 0
            steps = 0
            with tqdm(total=int(self._num_users // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._num_users, self._batch_size):
                    steps += 1

                    if self._total_anneal_steps > 0:
                        anneal = min(self._anneal_cap, 1. * self._update_count / self._total_anneal_steps)
                    else:
                        anneal = self._anneal_cap

                    loss += self._model.train_step(batch, anneal)
                    t.set_postfix({'loss': f'{loss.numpy()/steps:.5f}'})
                    t.update()
                    self._update_count += 1

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

    def restore_weights(self, it):
        if self._restore_epochs == it:
            try:
                self._model.load_weights(self._saving_filepath)
                print(f"Model correctly Restored at Epoch: {self._restore_epochs}")
                return True
            except Exception as ex:
                print(f"Error in model restoring operation! {ex}")
        return False

    # def get_recommendations(self, k: int = 100):
    #     predictions_top_k = {}
    #     for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
    #         offset_stop = min(offset + self._batch_size, self._num_users)
    #         predictions = self._model.predict(self._data.sp_i_train[offset:offset_stop].toarray())
    #         v, i = self._model.get_top_k(predictions, self.get_train_mask(offset, offset_stop), k=k)
    #         items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
    #                               for u_list in list(zip(i.numpy(), v.numpy()))]
    #         predictions_top_k.update(dict(zip(map(self._data.private_users.get,
    #                                               range(offset, offset_stop)), items_ratings_pair)))
    #     return predictions_top_k

