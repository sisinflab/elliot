"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from tqdm import tqdm
import numpy as np

from dataset.samplers import custom_sampler as cs
from evaluation.evaluator import Evaluator
from recommender import BaseRecommenderModel
from recommender.knowledge_aware.kaHFM_batch.kahfm_batch_model import KaHFM_model
from utils.write import store_recommendation
from recommender.knowledge_aware.kaHFM_batch.tfidf_utils import TFIDF
from recommender.recommender_utils_mixin import RecMixin

np.random.seed(42)


class KaHFMBatch(RecMixin, BaseRecommenderModel):

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

        self._ratings = self._data.train_dict
        self._sampler = cs.Sampler(self._data.i_train_dict)

        self._tfidf_obj = TFIDF(self._data.side_information_data.feature_map)
        self._tfidf = self._tfidf_obj.tfidf()
        self._user_profiles = self._tfidf_obj.get_profiles(self._ratings)

        self._user_factors = \
            np.zeros(shape=(len(self._data.users), len(self._data.features)))
        self._item_factors = \
            np.zeros(shape=(len(self._data.items), len(self._data.features)))

        for i, f_dict in self._tfidf.items():
            if i in self._data.items:
                for f, v in f_dict.items():
                    self._item_factors[self._data.public_items[i]][self._data.public_features[f]] = v

        for u, f_dict in self._user_profiles.items():
            for f, v in f_dict.items():
                self._user_factors[self._data.public_users[u]][self._data.public_features[f]] = v

        self._iteration = 0
        self.evaluator = Evaluator(self._data, self._params)
        if self._batch_size < 1:
            self._batch_size = self._num_users
        self._params.name = self.name

        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0001, None, None),
            ("_l_w", "l_w", "l_w", 0.005, None, None),
            ("_l_b", "l_b", "l_b", 0, None, None),
        ]
        self.autoset_params()

        self._factors = self._data.factors
        # self._learning_rate = self._params.lr
        # self._l_w = self._params.l_w
        # self._l_b = self._params.l_b

        self._model = KaHFM_model(self._user_factors,
                                  self._item_factors,
                                  self._params.lr,
                                  self._params.l_w,
                                  self._params.l_b)

        self._saving_filepath = f'{self._config.path_output_rec_weight}best-weights-{self.name}'

    @property
    def name(self):
        return "KaHFMBatch" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        best_metric_value = 0
        for it in range(self._epochs):
            self.restore_weights(it)
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                # for batch in zip(*self._sampler.step(self._data.transactions, self._batch_size)):
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            if not (it + 1) % self._validation_rate:
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                result_dict = self.evaluator.eval(recs)
                self._results.append(result_dict)

                print(f'Epoch {(it + 1)}/{self._epochs} loss {loss:.3f}')

                if self._results[-1][self._validation_k]["val_results"][self._validation_metric] > best_metric_value:
                    print("******************************************")
                    best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                    if self._save_weights:
                        self._model.save_weights(self._saving_filepath)
                    if self._save_recs:
                        store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

    # def restore_weights(self, it):
    #     if self._restore_epochs == it:
    #         try:
    #             self._model.load_weights(self._saving_filepath)
    #             print(f"Model correctly Restored at Epoch: {self._restore_epochs}")
    #             return True
    #         except Exception as ex:
    #             print(f"Error in model restoring operation! {ex}")
    #     return False

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset+self._batch_size, self._num_users)
            predictions = self._model.predict_batch(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, self.get_train_mask(offset, offset_stop), k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
        return predictions_top_k

    # def get_train_mask(self, start, stop):
    #     return np.where((self._data.sp_i_train[range(start, stop)].toarray() == 0), True, False)
    #
    # def get_loss(self):
    #     return -max([r[self._validation_metric] for r in self._results])
    #
    # def get_params(self):
    #     return self._params.__dict__
    #
    # def get_results(self):
    #     val_max = np.argmax([r[self._validation_metric] for r in self._results])
    #     return self._results[val_max]
    #
    # def get_statistical_results(self):
    #     val_max = np.argmax([r[self._validation_metric] for r in self._results])
    #     return self._statistical_results[val_max]
