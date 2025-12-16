from typing import Union
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
from torch.utils.data import DataLoader

from elliot.evaluation.evaluator import Evaluator
from elliot.recommender.early_stopping import EarlyStopping

from elliot.utils.config import TrainerConfig
from elliot.utils.folder import build_model_folder
from elliot.utils.write import store_recommendation
from elliot.utils import logging


class AbstractTrainer(ABC):
    restore: bool = False
    validation_metric: Union[str, list] = None
    save_weights: bool = False
    save_recs: bool = False
    verbose: bool = True
    validation_rate: int = 1
    optimize_internal_loss: bool = False
    epochs: int = 1
    batch_size: int = None
    eval_batch_size: int = None
    seed: int = 42

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
            self.get_recs = self.get_recs_neg_eval
        else:
            self.get_recs = self.get_recs_full_eval

        # Validate and assign meta parameters
        self.set_params(meta=True)

        # Validation metric
        _cutoff_k = getattr(data.config.evaluation, "cutoffs", [data.config.top_k])
        _cutoff_k = _cutoff_k if isinstance(_cutoff_k, list) else [_cutoff_k]
        _first_metric = data.config.evaluation.simple_metrics[0] if data.config.evaluation.simple_metrics else ""
        _default_validation_k = _cutoff_k[0]

        if self.validation_metric is None:
            self.validation_metric = _first_metric + "@" + str(_default_validation_k)

        validation_metric = self.validation_metric.split("@")

        if validation_metric[0].lower() not in [m.lower()
                                                      for m in data.config.evaluation.simple_metrics]:
            raise Exception("Validation metric must be in the list of simple metrics")

        self._validation_k = int(validation_metric[1]) if len(validation_metric) > 1 else _cutoff_k[0]

        if self._validation_k not in _cutoff_k:
            raise Exception("Validation cutoff must be in general cutoff values")

        self.validation_metric = validation_metric[0]

        # Early stopping
        self._early_stopping = EarlyStopping(SimpleNamespace(**getattr(self._params, "early_stopping", {})),
                                             self.validation_metric, self._validation_k, _cutoff_k,
                                             data.config.evaluation.simple_metrics)

        # Validate and assign other parameters
        self.set_params()

        if self.epochs < self.validation_rate:
            raise Exception(f"The first validation epoch ({self.validation_rate}) "
                            f"is later than the overall number of epochs ({self.epochs}).")

        if self.batch_size is None:
            self.batch_size = self._data.batch_size
        else:
            self._data.batch_size = self.batch_size

        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size

        # Set seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Logger
        package_name = inspect.getmodule(model_class).__package__
        rec_name = f"external.{model_class.__name__}" if "external" in package_name else model_class.__name__
        self.logger = logging.get_logger_model(rec_name, pylog.CRITICAL if self._config.config_test else pylog.DEBUG)

        # Model
        self._model = model_class(data, params, self.seed, self.logger)

        # Further parameters
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

    def set_params(self, meta: bool = False):
        """Validate and set object parameters.

        Args:
            meta (bool): If True assign metadata fields, otherwise training fields.
        """
        param_ns = self._params if not meta else self._params.meta
        config = TrainerConfig(**vars(param_ns))
        for name, val in config.get_validated_params(meta=meta).items():
            setattr(self, name, val)

    @property
    def name(self):
        return self._model.name + f"_{self.get_base_params_shortcut()}" + self._model.name_param

    def get_base_params_shortcut(self):
        return "_".join([str(k) + "=" + str(v).replace(".", "$") for k, v in
                         dict({"seed": self.seed,
                               "epochs": self.epochs,
                               "batch_size": self.batch_size,
                               "eval_batch_size": self.eval_batch_size}).items()
                         ])

    # def get_model_params_shortcut(self):
    #     return "_".join(
    #         [str(p[2])+"="+ str(p[5](getattr(self._model, p[0]))
    #                             if p[5] else getattr(self._model, p[0])).replace(".", "$")
    #          for p in self._model.params_list]
    #     )

    def train(self):
        if self.restore:
            return self.restore_weights()

        self.logger.info(
            "Loaded training dataset",
            extra={"context": {"transactions": self._data.transactions}}
        )
        training_dataloader = self._model.get_training_dataloader()

        if not isinstance(training_dataloader, DataLoader):
            self.verbose = False

        for it in self.iterate(self.epochs):
            self.logger.debug(
                "Starting iteration",
                extra={"context": {"iteration": it + 1, "epochs": self.epochs}}
            )
            start = time.perf_counter()
            loss = self._train_epoch(it, training_dataloader)
            end = time.perf_counter()
            self.logger.debug(
                "Completed iteration",
                extra={"context": {"iteration": it + 1, "duration_sec": end - start}}
            )
            if not (it + 1) % self.validation_rate:
                self.evaluate(it, loss / (it + 1))

    def evaluate(self, it=0, loss=0):
        recs = self.get_recs(self.evaluator.get_needed_recommendations())
        result_dict = self.evaluator.eval(recs)

        self._losses.append(loss)

        self._results.append(result_dict)

        # if it is not None:
        self.logger.debug(f'Epoch {(it + 1)}/{self.epochs} loss {loss / (it + 1):.5f}')
        # else:
        #    self.logger.info(f'Finished')

        if self.save_recs:
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
            best_val = self._results[-1][self._validation_k]["val_results"][self.validation_metric]
            self.best_metric_value = best_val
            self.logger.info(
                "Recorded best validation result",
                extra={"context": {"metric": self.validation_metric, "value": best_val, "iteration": it + 1}}
            )
            if self.save_weights:
                if hasattr(self, "_model"):
                    self._model.save_weights(self._saving_filepath)
                else:
                    self.logger.warning("Saving weights FAILED. No model to save.")

    def get_recs_full_eval(self, k: int = 100, *args):
        preds_test, preds_val = {}, {}
        dataloader = self._data.full_eval_dataloader(self.eval_batch_size)

        iter_data = tqdm(
            dataloader,
            desc="Full eval",
            total=len(dataloader),
            leave=False
        )

        for users in iter_data:
            train_batch = self._data.sp_i_train_ratings[users.tolist()]

            recs = self._compute_batch_recs(k=k, user_indices=users, train_batch=train_batch)

            preds_test.update(recs)
            preds_val.update(recs)

        return preds_val, preds_test

    def get_recs_neg_eval(self, k: int = 100, *args):
        preds_test, preds_val = {}, {}
        dataloader = self._data.neg_eval_dataloader(self.eval_batch_size)

        iter_data = tqdm(
            dataloader,
            desc="Neg eval",
            total=len(dataloader),
            leave=False
        )

        for users, val_items, test_items in iter_data:
            # Test
            recs_test = self._compute_batch_recs(k=k, user_indices=users, item_indices=test_items)

            # Validation
            if val_items is not None:
                recs_val = self._compute_batch_recs(k=k, user_indices=users, item_indices=val_items)
            else:
                recs_val = recs_test

            preds_test.update(recs_test)
            preds_val.update(recs_val)

        return preds_val, preds_test

    def _compute_batch_recs(self, k, user_indices, item_indices=None, train_batch=None):
        """Common logic for computing top-k recommendations."""
        if item_indices is not None:
            preds = self._model.predict_sampled(user_indices, item_indices)
            mask = item_indices == -1
        else:
            preds = self._model.predict_full(user_indices)
            mask = train_batch.nonzero()

        v, i = self._get_top_k(preds, k, mask, item_indices)
        recs_dict = self._get_recs_dict(v, i, user_indices)

        return recs_dict

    def _get_recs_dict(self, values, item_indices, user_indices):
        pr_users, pr_items = self._data.get_inverse_mappings()
        mapped_items = np.array(pr_items)[item_indices]
        mat = [[*zip(item, val)] for item, val in zip(mapped_items, values)]
        proc_batch = dict(zip([pr_users[u_i] for u_i in user_indices], mat))
        return proc_batch

    def _get_top_k(self, users_recs, k, mask, item_indices=None):
        users_recs[mask] = -torch.inf
        v, i = torch.topk(users_recs, k=k, sorted=True)

        if item_indices is not None:
            i = item_indices.gather(1, i)

        return v.numpy(), i.numpy()

    def restore_weights(self):
        try:
            self._model.load_weights(self._saving_filepath)
            self.logger.info(
                "Model restored from disk",
                extra={"context": {"path": self._saving_filepath}}
            )
            self.evaluate()
            return True
        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

    def get_loss(self):
        if self.optimize_internal_loss:
            return min(self._losses)
        else:
            return -max([r[self._validation_k]["val_results"][self.validation_metric] for r in self._results])

    def get_params(self):
        return self._params.__dict__

    def get_results(self):
        return self._results[self.get_best_arg()]

    def get_best_arg(self):
        if self.optimize_internal_loss:
            val_results = np.argmin(self._losses)
        else:
            val_results = np.argmax(
                [r[self._validation_k]["val_results"][self.validation_metric] for r in self._results])
        return val_results

    def iterate(self, epochs):
        for iteration in range(epochs):
            if self._early_stopping.stop(self._losses[:], self._results):
                self.logger.info(f"Met Early Stopping conditions: {self._early_stopping}")
                break
            else:
                yield iteration

    @abstractmethod
    def _train_epoch(self, it, dataloader, *args):
        raise NotImplementedError()

    #@staticmethod
    #def _batch_remove(original_str: str, char_list):
    #    for c in char_list:
    #        original_str = original_str.replace(c, "")
    #    return original_str


class Trainer(AbstractTrainer):
    def __init__(self, data, config, params, model_class):
        super().__init__(data, config, params, model_class)

    def _train_epoch(self, it, dataloader, *args):
        loss = 0
        steps = 0
        iter_ = tqdm(
            total=int(self._model.transactions // self.batch_size),
            desc="Training",
            disable=not self.verbose
        )
        with iter_ as t:
            for batch in dataloader:
                steps += 1
                loss += self._model.train_step(batch, *args)
                t.set_postfix({'loss': f'{loss / steps:.5f}'})
                t.update()
        return loss


class TraditionalTrainer(Trainer):
    def __init__(self, data, config, params, model_class):
        super().__init__(data, config, params, model_class)
        self.epochs = 1

    def _train_epoch(self, *args):
        self._model.initialize()
        return 0


class GeneralTrainer(AbstractTrainer):
    def __init__(self, data, config, params, model_class):
        super().__init__(data, config, params, model_class)
        self.optimizer = self._model.optimizer
        torch.manual_seed(self.seed)

    def _train_epoch(self, it, dataloader, *args):
        self._model.train()
        total_loss, steps = 0, 0
        iter_ = tqdm(
            total=int(self._model.transactions // self.batch_size),
            desc="Training",
            disable=not self.verbose
        )
        with iter_ as t:
            for batch in dataloader:
                steps += 1
                self.optimizer.zero_grad()
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
