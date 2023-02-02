
from datetime import datetime
import time
import numpy as np
import random
import argparse


from .GraphMaker import GraphMaker
"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import torch
import os
from tqdm import tqdm
import json

# from .custom_sampler import Sampler
from elliot.utils.write import store_recommendation

from elliot.recommender import BaseRecommenderModel
# from .BPRMFModel import BPRMFModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.base_recommender_model import init_charger


class CDRIB(RecMixin, BaseRecommenderModel):
    r"""
    Batch Bayesian Personalized Ranking with Matrix Factorization

    For further details, please refer to the `paper <https://arxiv.org/abs/1205.2618.pdf>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        l_w: Regularization coefficient for latent factors

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        BPRMF:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 10
          lr: 0.001
          l_w: 0.1
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a BPR-MF instance.
        (see https://arxiv.org/pdf/1205.2618 for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w]: regularization,
                                      lr: learning rate}
        """

        self._params_list = [
            ("_feature_dim", "feature_dim", "feature_dim", 128, int, None),
            ("_hidden_dim", "hidden_dim", "hidden_dim", 128, int, None),
            ("_gnn_layers", "gnn_layers", "gnn_layers", 3, int, None),
            ("_dropout", "dropout", "dropout", 0.3, float, None),
            ("_learning_rate", "lr", "lr", 0.001, float, None),
            ("_leakey", "leakey", "leakey", 0.1, float, None),
            ("_lambda", "lambda", "lambda", 0.9, float, None),
            ("_bce", "bce", "bce", True, bool, None),
            ("_margin", "margin", "margin", 0.3, float, None),
            ("_beta", "beta", "beta", 1.5, float, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        source_G = GraphMaker(opt, self._ratings)

        # self._sampler = Sampler(self._data.i_train_dict, self._seed)
        #
        # self._model = BPRMFModel(self._num_users,
        #                          self._num_items,
        #                          self._learning_rate,
        #                          self._factors,
        #                          self._l_w,
        #                          self._seed)

    @property
    def name(self):
        return "CDRIB" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

        with open('./results/{0}/performance/'.format(self._config.dataset) + 'freq_users.json', 'w') as f:
            json.dump(self._sampler.freq_users, f)
        with open('./results/{0}/performance/'.format(self._config.dataset) + 'freq_items.json', 'w') as f:
            json.dump(self._sampler.freq_items, f)

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss / (it + 1):.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._saving_filepath)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False















# class CrossDomainRecommenderInformationBottleneck(RecMixin, BaseRecommenderModel):
#
#     #Definisco il metodo init per inizializzare tutti i parametri del modello
#     @init_charger
#     def __init__(self):
#         #definisco la lista dei parametri settandoli automaticamente
#         self._params_list()
#         self.autoset_params()
#
#     def seed_everything(seed=1111):
#         random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         np.random.seed(seed)
#         os.environ['PYTHONHASHSEED'] = str(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
#     # make parser, args and opt
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     opt = vars(args)
#
#     def train(self, num_epoch, batch_size, user_batch_size, log_step, log, save_epoch, save_dir, id, seed, load,
#               model_file, test_sample_number):
#         for epoch in range(1, opt['num_epoch'] + 1):
#             train_loss = 0
#             start_time = time.time()
#             for i, batch in enumerate(train_batch):
#             global_step += 1
#            loss = trainer.reconstruct_graph(batch, source_UV, source_VU, target_UV, target_VU, source_adj, target_adj, epoch)
#            train_loss += loss
#
#         duration = time.time() - start_time
#         train_loss = train_loss/len(train_batch)
#         print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
#                                     opt['num_epoch'], train_loss, duration, current_lr))
#
#         if epoch<10 or epoch % 5:
#             continue
#
#      eval model
#     print("Evaluating on dev set...")
#     trainer.model.eval()
#
#     trainer.evaluate_embedding(source_UV, source_VU, target_UV, target_VU, source_adj, target_adj,epoch)
#
#     def predict(dataloder, choose):
#         MRR = 0.0
#         NDCG_1 = 0.0
#         NDCG_5 = 0.0
#         NDCG_10 = 0.0
#         HT_1 = 0.0
#         HT_5 = 0.0
#         HT_10 = 0.0
#
#         valid_entity = 0.0
#         for i, batch in enumerate(dataloder):
#             if choose:
#                 predictions = trainer.source_predict(batch)
#             else :
#                 predictions = trainer.target_predict(batch)
#             for pred in predictions:
#                 rank = (-pred).argsort().argsort()[0].item()
#
#                 valid_entity += 1
#                 MRR += 1 / (rank + 1)
#                 if rank < 1:
#                     NDCG_1 += 1 / np.log2(rank + 2)
#                     HT_1 += 1
#                 if rank < 5:
#                     NDCG_5 += 1 / np.log2(rank + 2)
#                     HT_5 += 1
#                 if rank < 10:
#                     NDCG_10 += 1 / np.log2(rank + 2)
#                     HT_10 += 1
#                 if valid_entity % 100 == 0:
#                     print('.', end='')
#         s_mrr = MRR / valid_entity
#         s_ndcg_5 = NDCG_5 / valid_entity
#         s_ndcg_10 = NDCG_10 / valid_entity
#         s_hr_1 = HT_1 / valid_entity
#         s_hr_5 = HT_5 / valid_entity
#         s_hr_10 = HT_10 / valid_entity
#
#        return s_mrr, s_ndcg_5, s_ndcg_10, s_hr_1, s_hr_5, s_hr_10
#
#     s_mrr, s_ndcg_5, s_ndcg_10, s_hr_1, s_hr_5, s_hr_10 = predict(source_valid_batch, 1)
#     t_mrr, t_ndcg_5, t_ndcg_10, t_hr_1, t_hr_5, t_hr_10 = predict(target_valid_batch, 0)
#
#     print("\nsource: \t{:.6f}\t{:.4f}\t{:.4f}\t{:.6f}\t{:.4f}\t{:.4f}".format(s_mrr, s_ndcg_5, s_ndcg_10, s_hr_1, s_hr_5, s_hr_10))
#     print("target: \t{:.6f}\t{:.4f}\t{:.4f}\t{:.6f}\t{:.4f}\t{:.4f}".format(t_mrr, t_ndcg_5, t_ndcg_10, t_hr_1, t_hr_5, t_hr_10))
#
#     s_dev_score = s_mrr
#     t_dev_score = t_mrr
#     if s_dev_score > max(s_dev_score_history):
#         print("source best!")
#         s_mrr, s_ndcg_5, s_ndcg_10, s_hr_1, s_hr_5, s_hr_10 = predict(source_test_batch, 1)
#         print("\nsource: \t{:.6f}\t{:.4f}\t{:.4f}\t{:.6f}\t{:.4f}\t{:.4f}".format(s_mrr, s_ndcg_5, s_ndcg_10, s_hr_1, s_hr_5, s_hr_10))
#
#     if t_dev_score > max(t_dev_score_history):
#         print("target best!")
#         t_mrr, t_ndcg_5, t_ndcg_10, t_hr_1, t_hr_5, t_hr_10 = predict(target_test_batch, 0)
#         print("target: \t{:.6f}\t{:.4f}\t{:.4f}\t{:.6f}\t{:.4f}\t{:.4f}".format(t_mrr, t_ndcg_5, t_ndcg_10, t_hr_1, t_hr_5, t_hr_10))

   #
   #  file_logger.log(
   #      "{}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, s_dev_score, max([s_dev_score] + s_dev_score_history)))
   #
   #  print(
   #      "epoch {}: train_loss = {:.6f}, source_hit = {:.4f}, source_ndcg = {:.4f}, target_hit = {:.4f}, target_ndcg = {:.4f}".format(
   #          epoch, \
   #          train_loss, s_hr_10, s_ndcg_10, t_hr_10, t_ndcg_10))
   #
   #
   # # save
   #  model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
   #  if epoch == 1 or s_dev_score > max(s_dev_score_history):
   #       copyfile(model_file, model_save_dir + '/best_model.pt')
   #      print("new best model saved.")
   #
   #  # lr schedule
   #  if len(s_dev_score_history) > opt['decay_epoch'] and s_dev_score <= s_dev_score_history[-1] and \
   #          opt['optim'] in ['sgd', 'adagrad', 'adadelta', 'adam']:
   #      current_lr *= opt['lr_decay']
   #      trainer.update_lr(current_lr)
   #
   #  s_dev_score_history += [s_dev_score]
   #  t_dev_score_history += [t_dev_score]
   #  print("")
   #
   #  def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
   #      #extra
   #  def get_recommendations(self, k: int = 100):
   #      #extra
   #
   #

