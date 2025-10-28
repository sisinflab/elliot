import inspect
import random
import logging as pylog
import os
import time

import torch
import numpy as np
import scipy.sparse as sp
from abc import ABC, abstractmethod
from tqdm import tqdm
from types import SimpleNamespace

from elliot.dataset.samplers.base_sampler import FakeSampler
from elliot.evaluation.evaluator import Evaluator
from elliot.recommender.early_stopping import EarlyStopping
from elliot.utils.folder import build_model_folder
from elliot.utils.write import store_recommendation
from elliot.utils import logging


class AbstractTrainer(ABC):
    def __init__(self, data, config, params, model_class, *args, **kwargs):
        """
        This class represents a recommender model. You can load a pretrained model
        by specifying its checkpoint path and use it for training/testing purposes.

        Args:
            data: data loader object
            params: dictionary with all parameters
        """
        self._data = data
        self._config = config
        self._params = params

        if hasattr(data.config, "negative_sampling"):
            self._mask_func = self._negative_sampling_eval
        else:
            self._mask_func = self._full_rank_eval

        # Base params
        self._restore = getattr(self._params.meta, "restore", False)

        _cutoff_k = getattr(data.config.evaluation, "cutoffs", [data.config.top_k])
        _cutoff_k = _cutoff_k if isinstance(_cutoff_k, list) else [_cutoff_k]
        _first_metric = data.config.evaluation.simple_metrics[0] if data.config.evaluation.simple_metrics else ""
        _default_validation_k = _cutoff_k[0]
        self._validation_metric = getattr(self._params.meta, "validation_metric",
                                          _first_metric + "@" + str(_default_validation_k)).split("@")
        if self._validation_metric[0].lower() not in [m.lower()
                                                      for m in data.config.evaluation.simple_metrics]:
            raise Exception("Validation metric must be in the list of simple metrics")

        self._validation_k = int(self._validation_metric[1]) if len(self._validation_metric) > 1 else _cutoff_k[0]
        if self._validation_k not in _cutoff_k:
            raise Exception("Validation cutoff must be in general cutoff values")

        self._validation_metric = self._validation_metric[0]
        self._save_weights = getattr(self._params.meta, "save_weights", False)
        self._save_recs = getattr(self._params.meta, "save_recs", False)
        self._verbose = getattr(self._params.meta, "verbose", True)
        self._validation_rate = getattr(self._params.meta, "validation_rate", 1)
        self._optimize_internal_loss = getattr(self._params.meta, "optimize_internal_loss", False)
        self._epochs = int(getattr(self._params, "epochs", 1))
        self._seed = getattr(self._params, "seed", 42)
        self._early_stopping = EarlyStopping(SimpleNamespace(**getattr(self._params, "early_stopping", {})),
                                             self._validation_metric, self._validation_k, _cutoff_k,
                                             data.config.evaluation.simple_metrics)
        self._iteration = 0
        if self._epochs < self._validation_rate:
            raise Exception(f"The first validation epoch ({self._validation_rate}) "
                            f"is later than the overall number of epochs ({self._epochs}).")

        self._batch_size = (
            self._params.batch_size if getattr(self._params, "batch_size", 0) > 0
            else self._data.batch_size
        )
        self._data.batch_size = self._batch_size

        np.random.seed(self._seed)
        random.seed(self._seed)
        #self._nprandom = np.random
        #self._random = random

        # Logger
        package_name = inspect.getmodule(model_class).__package__
        rec_name = f"external.{model_class.__name__}" if "external" in package_name else model_class.__name__
        self.logger = logging.get_logger_model(rec_name, pylog.CRITICAL if self._config.config_test else pylog.DEBUG)

        # Model
        self._model = model_class(data, params, self._seed, self.logger)

        # Sampler
        self._sampler = self._model.sampler
        self._sampler.batch_size = self._batch_size
        if isinstance(self._sampler, FakeSampler):
            self._verbose = False
        # self._sampler.events = data.transactions

        # Other params
        self._num_items = self._data.num_items
        self._num_users = self._data.num_users

        self.best_metric_value = 0

        self._losses = []
        self._results = []
        self._params_list = []

        # Evaluator
        self.evaluator = Evaluator(self._data, self._params)

        # Saving file
        self._params.name = self.name
        build_model_folder(self._config.path_output_rec_weight, self.name)
        self._saving_filepath = os.path.abspath(
            os.sep.join([self._config.path_output_rec_weight, self.name, f"best-weights-{self.name}"])
        )

    @property
    def name(self):
        return self._model.name + f"_{self.get_base_params_shortcut()}" + f"_{self.get_model_params_shortcut()}"

    def get_base_params_shortcut(self):
        return "_".join([str(k) + "=" + str(v).replace(".", "$") for k, v in
                         dict({"seed": self._seed,
                               "e": self._epochs,
                               "bs": self._batch_size}).items()
                         ])

    def get_model_params_shortcut(self):
        return "_".join(
            [str(p[2])+"="+ str(p[5](getattr(self._model, p[0]))
                                if p[5] else getattr(self._model, p[0])).replace(".", "$")
             for p in self._model.params_list]
        )

    @abstractmethod
    def _train_epoch(self, it, *args):
        pass

    def train(self):
        if self._restore:
            return self.restore_weights()

        print(f"Transactions: {self._data.transactions}")

        for it in self.iterate(self._epochs):
            print(f"\n********** Iteration: {it + 1}")
            start = time.perf_counter()
            loss = self._train_epoch(it)
            end = time.perf_counter()
            print(f"Duration: {end - start}")
            if not (it + 1) % self._validation_rate:
                self.evaluate(it, loss / (it + 1))

    def evaluate(self, it=0, loss=0):
        recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
        result_dict = self.evaluator.eval(recs)

        self._losses.append(loss)

        self._results.append(result_dict)

        # if it is not None:
        self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss / (it + 1):.5f}')
        # else:
        #    self.logger.info(f'Finished')

        if self._save_recs:
            self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
            # if it is not None:
            store_recommendation(recs[1], os.path.abspath(
                os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
            # else:
            #    store_recommendation(recs[1], os.path.abspath(
            #        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

        if (len(self._results) - 1) == self.get_best_arg():
            # if it is not None:
            self._params.best_iteration = it + 1
            self.logger.info("******************************************")
            self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
            if self._save_weights:
                if hasattr(self, "_model"):
                    self._model.save_weights(self._saving_filepath)
                else:
                    self.logger.warning("Saving weights FAILED. No model to save.")

    @abstractmethod
    def get_recommendations(self, k, *args):
        pass

    def process_protocol(self, k, masks, predictions, start, stop):
        val_mask, test_mask = masks
        test_recs = self.get_single_recommendation(k, test_mask, predictions, start, stop)
        val_recs = self.get_single_recommendation(k, val_mask, predictions, start, stop) if val_mask else test_recs
        return val_recs, test_recs

    def get_single_recommendation(self, k, mask, predictions, start, stop):
        v, i = self._get_top_k(self._mask_func(mask, predictions), k)
        mapped_items = np.array(self._data.private_items)[i]
        mat = [[*zip(item, val)] for item, val in zip(mapped_items, v)]
        proc_batch = dict(zip(self._data.private_users[start:stop], mat))
        return proc_batch

    @abstractmethod
    def _full_rank_eval(self, mask, preds):
        raise NotImplementedError()

    @abstractmethod
    def _negative_sampling_eval(self, mask, preds):
        raise NotImplementedError()

    @abstractmethod
    def _get_top_k(self, users_recs, k):
        raise NotImplementedError()

    def restore_weights(self):
        try:
            self._model.load_weights(self._saving_filepath)
            print(f"Model correctly Restored")
            self.evaluate()
            return True
        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

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
            val_results = np.argmax(
                [r[self._validation_k]["val_results"][self._validation_metric] for r in self._results])
        return val_results

    def iterate(self, epochs):
        for iteration in range(epochs):
            if self._early_stopping.stop(self._losses[:], self._results):
                self.logger.info(f"Met Early Stopping conditions: {self._early_stopping}")
                break
            else:
                yield iteration

    #@staticmethod
    #def _batch_remove(original_str: str, char_list):
    #    for c in char_list:
    #        original_str = original_str.replace(c, "")
    #    return original_str


class Trainer(AbstractTrainer):
    def __init__(self, data, config, params, model_class):
        super().__init__(data, config, params, model_class)

    def _train_epoch(self, it, *args):
        loss = 0
        steps = 0
        self._sampler.initialize()
        iter_ = tqdm(
            total=int(self._model.transactions // self._batch_size),
            desc="Training",
            disable=not self._verbose
        )
        with iter_ as t:
            for batch in self._sampler.step():
                steps += 1
                loss += self._model.train_step(batch, *args)
                t.set_postfix({'loss': f'{loss / steps:.5f}'})
                t.update()
        return loss

    def get_recommendations(self, k: int = 100, *args):
        predictions_top_k_test = {}
        predictions_top_k_val = {}

        for (start, stop), masks in tqdm(self._data, desc="Processing batches", total=len(self._data)):
            predictions = self._model.predict(start, stop)
            recs_val, recs_test = self.process_protocol(k, masks, predictions, start, stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def _full_rank_eval(self, mask, preds):
        if isinstance(preds, sp.csr_matrix):
            preds = preds.toarray()
        preds[mask.nonzero()] = -np.inf
        return preds

    def _negative_sampling_eval(self, mask, preds):
        if isinstance(preds, sp.csr_matrix):
            preds = preds.multiply(mask).toarray()
        else:
            preds = np.multiply(preds, mask.toarray())
        return preds

    def _get_top_k(self, users_recs, k):
        index_ordered = np.argpartition(users_recs, -k, axis=1)[:, -k:]
        value_ordered = np.take_along_axis(users_recs, index_ordered, axis=1)
        local_top_k = np.take_along_axis(index_ordered, value_ordered.argsort(axis=1)[:, ::-1], axis=1)
        value_sorted = np.take_along_axis(users_recs, local_top_k, axis=1)
        return value_sorted, local_top_k


class TraditionalTrainer(Trainer):
    def __init__(self, data, config, params, model_class):
        super().__init__(data, config, params, model_class)
        self._epochs = 1

    def _train_epoch(self, *args):
        self._model.initialize()
        return 0


class GeneralTrainer(AbstractTrainer):
    def __init__(self, data, config, params, model_class):
        super().__init__(data, config, params, model_class)
        self.optimizer = self._model.optimizer
        torch.manual_seed(self._seed)

    def _train_epoch(self, it, *args):
        self._model.train()
        total_loss, steps = 0, 0
        self._sampler.initialize()
        iter_ = tqdm(
            total=int(self._model.transactions // self._batch_size),
            desc="Training",
            disable=not self._verbose
        )
        with iter_ as t:
            for batch in self._sampler.step():
                steps += 1
                self.optimizer.zero_grad()
                batch = tuple(torch.tensor(b, dtype=torch.int64) for b in batch)
                res = self._model.train_step(batch, steps, *args)
                loss, inputs = res if isinstance(res, tuple) else (res, None)
                loss.backward(inputs=inputs)
                total_loss += loss.detach().cpu().numpy()
                self.optimizer.step()
                t.set_postfix({'loss': f'{total_loss / steps:.5f}'})
                t.update()
        return total_loss

    @torch.no_grad()
    def evaluate(self, it=0, loss=0):
        self._model.eval()
        super().evaluate(it, loss)

    def get_recommendations(self, k: int = 100, *args):
        predictions_top_k_test = {}
        predictions_top_k_val = {}

        for (start, stop), masks in tqdm(self._data, desc="Processing batches", total=len(self._data)):
            predictions = self._model.predict(start, stop)
            recs_val, recs_test = self.process_protocol(k, masks, predictions, start, stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def _full_rank_eval(self, mask, preds):
        preds[mask.nonzero()] = -np.inf
        return preds

    def _negative_sampling_eval(self, mask, preds):
        preds = preds * torch.tensor(mask.toarray())
        return preds

    def _get_top_k(self, users_recs, k):
        v, i = torch.topk(users_recs, k=k, sorted=True)
        return v.numpy(), i.numpy()
