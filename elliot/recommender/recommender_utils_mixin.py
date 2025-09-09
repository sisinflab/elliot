import os

import numpy as np
from tqdm import tqdm

from elliot.utils.write import store_recommendation


class RecMixin(object):
    def __init__(self):
        if hasattr(self, "_model"):
            self._model.apply_mask = self._make_item_mask_function()

    def _make_item_mask_function(self):
        if self._negative_sampling:
            def apply_mask(matrix, mask):
                return (matrix.multiply(mask)).toarray()
        else:
            def apply_mask(matrix, mask):
                result = matrix.copy().toarray()
                result[mask.nonzero()] = -np.inf
                return result
        return apply_mask

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

        for batch, masks in tqdm(self._data, desc="Processing batches", total=len(self._data)):
            #offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(batch.toarray())
            recs_val, recs_test = self.process_protocol(k, masks, predictions)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def process_protocol(self, k, masks, *args):
        val_mask, test_mask = masks
        test_recs = self.get_single_recommendation(k, test_mask, *args)
        val_recs = self.get_single_recommendation(k, val_mask, *args) if val_mask else test_recs
        return val_recs, test_recs
        """if not val_mask:
            self._current_mask = test_mask
            recs = self.get_single_recommendation(neg_test_indices, k, *args)
            return recs, recs
        
            return self.get_single_recommendation(neg_val_indices, k, *args) if self._data.val_dict is not None else {}, \
                   self.get_single_recommendation(neg_test_indices, k, *args)"""

    def get_single_recommendation(self, k, mask, predictions, offset, offset_stop):
        pass
        #validated_mask = self.get_mask_portion(mask, offsets=(offset, offset_stop))
        #v, i = self._model.get_top_k(predictions, mask.toarray(), k=k)
        #items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
        #                     for u_list in list(zip(i.numpy(), v.numpy()))]
        #return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    """def get_user_mask(self, mask, user_id):
        user_mask = mask[user_id].toarray().flatten()
        return user_mask if not self._inverted else ~user_mask

    def get_mask_portion(self, mask, offsets):
        mask_portion = mask[offsets[0]:offsets[1]].toarray()
        return mask_portion if not self._inverted else ~mask_portion"""

    def restore_weights(self):
        try:
            self._model.load_weights(self._saving_filepath)
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False

    """def get_candidate_mask(self, validation=False):
        if self._negative_sampling:
            if validation:
                #self._inverted = self._data.inverted['val_mask']
                return self._data.val_mask
            else:
                #self._inverted = self._data.inverted['test_mask']
                return self._data.test_mask
        else:
            #self._inverted = self._data.inverted['all_unrated_mask']
            return None #self._data.all_unrated_mask"""

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



