import logging
import os

import numpy as np

from config.configs import *
from dataset.dataset import DataSet
from dataset.samplers import sparse_sampler as sp
from evaluation.evaluator import Evaluator
from recommender import BaseRecommenderModel
from utils.read import find_checkpoint
from utils.write import save_obj, store_recommendation
import random

from .multi_dae_utils import DenoisingAutoEncoder
from .data_model import DataModel

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class MultiDAE(BaseRecommenderModel):

    def __init__(self, config, params, *args, **kwargs):
        """
        """
        super().__init__(config, params, *args, **kwargs)
        np.random.seed(42)
        random.seed(0)

        self._data = DataSet(config, params)
        self._config = config
        self._params = params
        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random
        self._random_p = random
        self._num_iters = self._params.epochs
        self._save_weights = self._params.save_weights
        self._save_recs = self._params.save_recs

        self._restore_epochs = -1
        self._ratings = self._data.train_dataframe_dict
        self._datamodel = DataModel(self._data.train_dataframe, self._ratings, self._random)
        self._sampler = sp.Sampler(self._datamodel.sp_train, self._random_p)
        self._iteration = 0
        self.evaluator = Evaluator(self._data)
        self._results = []

        self._maxtpu = max([len(items) for items in self._data.train_dataframe_dict.values()])

        ######################################

        self._params.name = self.name

        self._intermediate_dim = self._params.intermediate_dim
        self._latent_dim = self._params.latent_dim

        self._lambda = self._params.reg_lambda
        self._learning_rate = self._params.lr
        self._dropout_rate = 1. - self._params.dropout_pkeep

        self._verbose = self._params.verbose

        self._model = DenoisingAutoEncoder(self._num_items,
                                           self._intermediate_dim,
                                           self._latent_dim,
                                           self._learning_rate,
                                           self._dropout_rate,
                                           self._lambda)

        self._saving_filepath = f'{self._config.path_output_rec_weight}best-weights-{self.name}'
        self._train_mask = np.where((self._datamodel.sp_train.toarray() == 0), True, False)

    @property
    def name(self):
        return "MultiDAE" \
               + "_lr:" + str(self._params.lr) \
               + "-e:" + str(self._params.epochs) \
               + "-idim:" + str(self._params.intermediate_dim) \
               + "-ldim:" + str(self._params.latent_dim) \
               + "-bs:" + str(self._params.batch_size) \
               + "-dpk:" + str(self._params.dropout_pkeep) \
               + "-lmb" + str(self._params.reg_lambda)

    def train(self):

        best_ndcg = 0

        for it in range(self._num_iters):
            self.restore_weights(it)
            loss = 0
            steps = 0
            for batch in self._sampler.step(self._num_users, self._params.batch_size):
                steps += 1
                loss += self._model.train_step(batch.toarray())

            if not (it + 1) % self._params.verbose:
                recs = self.get_recommendations(self._config.top_k)
                self._results.append(self.evaluator.eval(recs))
                print(f'Epoch {(it + 1)}/{self._num_iters} loss {loss:.3f}')

                if self._results[-1]['nDCG'] > best_ndcg:
                    print("******************************************")
                    best_ndcg = self._results[-1]['nDCG']
                    # best_epoch = it
                    # self._model.saver_ckpt.save(f'''{self._config.path_output_rec_weight}best-weights-{best_epoch}-{self._learning_rate}-{self.__class__.__name__}''')
                    if self._save_weights:
                        self._model.save_weights(self._saving_filepath)
                    if self._save_recs:
                        store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

    def restore_weights(self, it):
        if self._restore_epochs == it:
            try:
                # checkpoint_file = find_checkpoint(weight_dir, self.restore_epochs, self.epochs,
                #                                   self.rec)
                self._model.load_weights(self._saving_filepath)
                # self.saver_ckpt.restore(checkpoint_file)

                print(f"Model correctly Restored at Epoch: {self._restore_epochs}")
                return True
            except Exception as ex:
                print(f"Error in model restoring operation! {ex}")
        return False

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        preds = self._model.predict(self._datamodel.sp_train.toarray())
        v, i = self._model.get_top_k(preds, self._train_mask, k=k)
        items_ratings_pair = [
            list(
            zip(
                map(self._datamodel.private_items.get, u_list[0]), u_list[1]
            )
        )
                              for u_list in list(zip(i.numpy(), v.numpy()))
        ]
        predictions_top_k.update(dict(zip(map(self._datamodel.private_users.get,
                                              range(self._datamodel.sp_train.shape[0])), items_ratings_pair)))
        return predictions_top_k

    # def get_recommendations(self, k: int = 100):
    #     local_k = k + self._maxtpu
    #     predictions_top_k = {}
    #     preds = self._model.predict(self._datamodel.sp_train.toarray())
    #     v, i = self._model.get_top_k(preds, k=local_k)
    #     items_ratings_pair = [
    #         [(item, value) for item, value in
    #         zip(
    #             map(self._datamodel.private_items.get, u_list[0]), u_list[1]
    #         ) if item not in self._data.train_dataframe_dict.get(self._datamodel.private_users.get(u_index)).keys()
    #     ][:k]
    #                           for u_index, u_list in enumerate(list(zip(i.numpy(), v.numpy())))
    #     ]
    #     predictions_top_k.update(dict(zip(map(self._datamodel.private_users.get,
    #                                           range(self._datamodel.sp_train.shape[0])), items_ratings_pair)))
    #     return predictions_top_k

    def get_loss(self):
        return -max([r["nDCG"] for r in self._results])

    def get_params(self):
        return self._params.__dict__

    def get_results(self):
        val_max = np.argmax([r["nDCG"] for r in self._results])
        return self._results[val_max]
