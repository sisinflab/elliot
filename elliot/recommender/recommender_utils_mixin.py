import numpy as np
from tqdm import tqdm

from utils.write import store_recommendation


class RecMixin(object):

    def train(self):
        self.logger.critical("Test2")
        best_metric_value = 0

        for it in range(self._num_iters):
            self.restore_weights(it)
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy()/steps:.5f}'})
                    t.update()

            if not (it + 1) % self._validation_rate:
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                results, statistical_results, test_results, test_statistical_results = self.evaluator.eval(recs)
                self._results.append(results)
                self._statistical_results.append(statistical_results)
                self._test_results.append(results)
                self._test_statistical_results.append(statistical_results)

                print(f'Epoch {(it + 1)}/{self._num_iters} loss {loss/steps:.5f}')

                if self._results[-1][self._validation_metric] > best_metric_value:
                    print("******************************************")
                    best_metric_value = self._results[-1][self._validation_metric]
                    if self._save_weights:
                        self._model.save_weights(self._saving_filepath)
                    if self._save_recs:
                        store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(self._data.sp_i_train[offset:offset_stop].toarray())
            v, i = self._model.get_top_k(predictions, self.get_train_mask(offset, offset_stop), k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(map(self._data.private_users.get,
                                                  range(offset, offset_stop)), items_ratings_pair)))
        return predictions_top_k

    def restore_weights(self, it):
        if self._restore_epochs == it:
            try:
                self._model.load_weights(self._saving_filepath)
                print(f"Model correctly Restored at Epoch: {self._restore_epochs}")
                return True
            except Exception as ex:
                print(f"Error in model restoring operation! {ex}")
        return False

    def get_train_mask(self, start, stop):
        return np.where((self._data.sp_i_train[range(start, stop)].toarray() == 0), True, False)

    def get_loss(self):
        return -max([r[self._validation_metric] for r in self._results])

    def get_params(self):
        return self._params.__dict__

    def get_results(self):
        val_max = np.argmax([r[self._validation_metric] for r in self._results])
        return self._results[val_max]

    def get_test_results(self):
        val_max = np.argmax([r[self._validation_metric] for r in self._results])
        return self._test_results[val_max]

    def get_statistical_results(self):
        val_max = np.argmax([r[self._validation_metric] for r in self._results])
        return self._statistical_results[val_max]

    def get_test_statistical_results(self):
        val_max = np.argmax([r[self._validation_metric] for r in self._results])
        return self._test_statistical_results[val_max]
