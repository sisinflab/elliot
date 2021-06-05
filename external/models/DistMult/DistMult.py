"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from tqdm import tqdm

from .triple_sampler import TripleSampler as TS
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from .DistMult_model import DistMultModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.lp_evaluation.evaluator import LPEvaluator


class DistMult(RecMixin, BaseRecommenderModel):
    r"""

    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """

        """
        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.1, None, None),
            ("_factors", "factors", "f", 100, int, None),
            ("_F2_weight", "F2_weight", "F2_weight", 0, None, None),
            ("_N3_weight", "N3_weight", "N3_weight", 0.001, None, None),
            ("_input_type", "input_type", "intype", "standard", None, None),
            ("_loader", "loader", "load", "KGCompletion", None, None),
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict

        self._side = getattr(self._data.side_information, self._loader, None)

        self._lp_evaluator = LPEvaluator(self._side, self._config, self._params)

        self._sampler = TS(self._side, self._seed)

        if self._batch_size < 1:
            self._batch_size = self._num_users

        self._transactions_per_epoch = self._side.Xs.shape[0]

        self._model = DistMultModel(self._side,
                                    self._learning_rate,
                                    self._factors,
                                    self._F2_weight,
                                    self._N3_weight,
                                    self._input_type,
                                    self._seed)

    @property
    def name(self):
        return "DistMult" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._transactions_per_epoch // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()
            self.evaluate(it, loss/(it + 1))

    def evaluate(self, it = None, loss = 0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            lp_result_dict = self._lp_evaluator.eval(self._model)

            self._losses.append(loss)

            self._results.append(lp_result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss/(it + 1):.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")

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
            predictions = self._model.predict_batch(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test
