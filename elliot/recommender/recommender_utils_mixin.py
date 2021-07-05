import os

import numpy as np
from tqdm import tqdm

from elliot.utils.write import store_recommendation


class RecMixin(object):

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in range(self._num_iters):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch).numpy()
                    t.set_postfix({'loss': f'{loss/steps:.5f}'})
                    t.update()

            self.evaluate(it, loss/(it + 1))

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss/(it + 1):.5f}')
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
                        self._model.save_weights(self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")



    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(self._data.sp_i_train[offset:offset_stop].toarray())
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def process_protocol(self, k, *args):

        if not self._negative_sampling:
            recs = self.get_single_recommendation(self.get_candidate_mask(), k, *args)
            return recs, recs
        else:
            return self.get_single_recommendation(self.get_candidate_mask(validation=True), k, *args) if hasattr(self._data, "val_dict") else {}, \
                   self.get_single_recommendation(self.get_candidate_mask(), k, *args)

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.numpy(), v.numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def restore_weights(self):
        try:
            self._model.load_weights(self._saving_filepath)
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False

    def get_candidate_mask(self, validation=False):
        if self._negative_sampling:
            if validation:
                return self._data.val_mask
            else:
                return self._data.test_mask
        else:
            return self._data.allunrated_mask

    def get_loss(self):
        if self._optimize_internal_loss:
            return min(self._losses)
        else:
            return -max([r[self._validation_k]["val_results"][self._validation_metric] for r in self._results])

    def get_params(self):
        return self._params.__dict__

    def get_results(self):
        return self._results[self.get_best_arg()]

    def get_best_arg(self):
        if self._optimize_internal_loss:
            val_results = np.argmin(self._losses)
        else:
            val_results = np.argmax([r[self._validation_k]["val_results"][self._validation_metric] for r in self._results])
        return val_results

    def iterate(self, epochs):
        for iteration in range(epochs):
            if self._early_stopping.stop(self._losses[:], self._results):
                self.logger.info(f"Met Early Stopping conditions: {self._early_stopping}")
                break
            else:
                yield iteration



