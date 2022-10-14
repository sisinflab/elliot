import numpy as np
import torch
import os

from elliot.utils.write import store_recommendation
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .SVDGCNSModel import SVDGCNSModel


class SVDGCNS(RecMixin, BaseRecommenderModel):
    r"""
    SVD-GCN-S: A Simplified Graph Convolution Paradigm for Recommendation
    (Non-trainable version)

    Args:
        batch_size: Batch size
        beta: Beta parameter for weighting function
        req_vec: No description
        alpha: For the R normalization

    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_alpha", "alpha", "alpha", 1, float, None),
            ("_beta", "beta", "beta", 0.1, float, None),
            ("_req_vec", "req_vec", "req_vec", 90, int, None)
        ]
        self.autoset_params()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rate_matrix = torch.from_numpy(data.sp_i_train.todense())
        self.rate_matrix.to(self.device)

        if self._config.data_config.strategy == 'fixed':
            path, _ = os.path.split(self._config.data_config.train_path)
            if not (os.path.exists(path + '/svd_u.npy') or os.path.exists(path + '/svd_i.npy') or os.path.exists(
                    path + '/svd_value.npy')):
                self.logger.info(
                    f"Processing singular values as they haven't been calculated before on this dataset...")
                U, value, V = self.preprocess(path)
                self.logger.info(f"Processing end!")
            else:
                self.logger.info(f"Singular values have already been processed for this dataset!")
                value = torch.Tensor(np.load(path + r'/svd_value.npy'))
                U = torch.Tensor(np.load(path + r'/svd_u.npy'))
                V = torch.Tensor(np.load(path + r'/svd_v.npy'))
        else:
            raise NotImplementedError('The check when strategy is different from fixed has not been implemented yet!')

        self._model = SVDGCNSModel(
            req_vec=self._req_vec,
            u=U,
            value=value,
            v=V,
            beta=self._beta
        )

    @property
    def name(self):
        return "SVDGCNS" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def preprocess(self, dataset):
        D_u = self.rate_matrix.sum(1) + self._alpha
        D_i = self.rate_matrix.sum(0) + self._alpha

        for i in range(self._num_users):
            if D_u[i] != 0:
                D_u[i] = 1 / D_u[i].sqrt()

        for i in range(self._num_items):
            if D_i[i] != 0:
                D_i[i] = 1 / D_i[i].sqrt()

        # \tilde{R}
        rate_matrix = D_u.unsqueeze(1) * self.rate_matrix * D_i

        # free space
        del D_u, D_i

        U, value, V = torch.svd_lowrank(rate_matrix, q=400, niter=30)

        np.save(dataset + r'/svd_u.npy', U.cpu().numpy())
        np.save(dataset + r'/svd_v.npy', V.cpu().numpy())
        np.save(dataset + r'/svd_value.npy', value.cpu().numpy())

        return U, value, V

    def train(self):
        self.evaluate()

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self.get_users_rating(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_users_rating(self, batch_start, batch_stop):
        return (self._model.user_vector[batch_start: batch_stop].mm(self._model.item_vector.t())).sigmoid().to(
            self.device) - (self.rate_matrix[batch_start: batch_stop] * 1000).to(self.device)

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
