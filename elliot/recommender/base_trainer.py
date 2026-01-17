from typing import Union
import inspect
import random
import logging as pylog
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
from elliot.utils.read import Reader
from elliot.utils.write import Writer
from elliot.utils import logging

reader = Reader()
writer = Writer()


class AbstractTrainer(ABC):
    config: TrainerConfig

    def __init__(self, data, config, params, model_class, *args, **kwargs):
        """
        This class represents a recommender model. You can load a pretrained model
        by specifying its checkpoint path and use it for training/testing purposes.

        Args:
            data: data loader object
            params: dictionary with all parameters
        """
        self._data = data
        self.global_config = config

        # _params = params.copy()
        if hasattr(params, 'meta'):
            params.meta = vars(params.meta)
        self.config = TrainerConfig(**vars(params))

        # Logger
        package_name = inspect.getmodule(model_class).__package__
        rec_name = f"external.{model_class.__name__}" if "external" in package_name else model_class.__name__
        self.logger = logging.get_logger_model(
            rec_name,
            pylog.CRITICAL if self.global_config.config_test else pylog.DEBUG
        )

        # Model
        self.model = model_class(data, params, self.config.seed, self.logger)

        # Validate and assign meta parameters
        # self.set_params(meta=True)

        # Validation metric
        _cutoff_k = getattr(data.config.evaluation, "cutoffs", [data.config.top_k])
        _cutoff_k = _cutoff_k if isinstance(_cutoff_k, list) else [_cutoff_k]
        _first_metric = data.config.evaluation.simple_metrics[0] if data.config.evaluation.simple_metrics else ""
        _default_validation_k = _cutoff_k[0]

        validation_metric = self.config.meta.validation_metric
        if validation_metric is None:
            validation_metric = _first_metric + "@" + str(_default_validation_k)

        validation_metric = validation_metric.split("@")

        if validation_metric[0].lower() not in [m.lower()
                                                      for m in data.config.evaluation.simple_metrics]:
            raise Exception("Validation metric must be in the list of simple metrics")

        self._validation_k = int(validation_metric[1]) if len(validation_metric) > 1 else _cutoff_k[0]

        if self._validation_k not in _cutoff_k:
            raise Exception("Validation cutoff must be in general cutoff values")

        self.validation_metric = validation_metric[0]

        # Early stopping
        self._early_stopping = EarlyStopping(SimpleNamespace(**getattr(params, "early_stopping", {})),
                                             self.validation_metric, self._validation_k, _cutoff_k,
                                             data.config.evaluation.simple_metrics)

        # Validate and assign other parameters
        # self.set_params()

        if self.config.epochs < self.config.meta.validation_rate:
            raise Exception(f"The first validation epoch ({self.config.meta.validation_rate}) "
                            f"is later than the overall number of epochs ({self.config.epochs}).")

        if self.config.eval_batch_size is None:
            self.config.eval_batch_size = self.config.batch_size

        # Set seed
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        # Further parameters
        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self.config.name = self.name

        self.best_metric_value = 0

        self._losses = []
        self._results = []
        self._params_list = []

        # Evaluator
        self.evaluator = Evaluator(data, params)

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
        return self.model.name + f"_{self.get_base_params_shortcut()}" + self.model.name_param

    def get_base_params_shortcut(self):
        return "_".join([str(k) + "=" + str(v).replace(".", "$") for k, v in
                         dict({"seed": self.config.seed,
                               "epochs": self.config.epochs,
                               "batch_size": self.config.batch_size,
                               "eval_batch_size": self.config.eval_batch_size}).items()
                         ])

    # def get_model_params_shortcut(self):
    #     return "_".join(
    #         [str(p[2])+"="+ str(p[5](getattr(self._model, p[0]))
    #                             if p[5] else getattr(self._model, p[0])).replace(".", "$")
    #          for p in self._model.params_list]
    #     )

    def train(self):
        if self.config.meta.restore:
            return self.restore_weights()

        self.logger.info(
            "Loaded training dataset",
            extra={"context": {"transactions": self._data.transactions}}
        )
        training_dataloader = self.model.get_training_dataloader(self.config.batch_size)

        if not isinstance(training_dataloader, DataLoader):
            self.config.meta.verbose = False

        for it in self.iterate(self.config.epochs):
            self.logger.debug(
                "Starting iteration",
                extra={"context": {"iteration": it + 1, "epochs": self.config.epochs}}
            )
            start = time.perf_counter()
            loss = self._train_epoch(it, training_dataloader)
            end = time.perf_counter()
            self.logger.debug(
                "Completed iteration",
                extra={"context": {"iteration": it + 1, "duration_sec": end - start}}
            )
            if not (it + 1) % self.config.meta.validation_rate:
                self.evaluate(it, loss / (it + 1))

    def evaluate(self, it=0, loss=0):
        recs = self.get_recs(self.evaluator.get_needed_recommendations())
        result_dict = self.evaluator.eval(recs)

        self._losses.append(loss)

        self._results.append(result_dict)

        # if it is not None:
        self.logger.debug(f'Epoch {(it + 1)}/{self.config.epochs} loss {loss / (it + 1):.5f}')
        # else:
        #    self.logger.info(f'Finished')

        if self.config.meta.save_recs:
            self.logger.info(f"Writing recommendations at: {self.global_config.path_output_rec_result}")
            # if it is not None:
            writer.write_recommendation(
                recommendations=recs[1],
                save_folder=self.global_config.path_output_rec_result,
                model_name=self.name,
                it=it
            )
            # else:
            #    store_recommendation(recs[1], os.path.abspath(
            #        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

        if (len(self._results) - 1) == self.get_best_arg():
            # if it is not None:
            self.config.best_iteration = it + 1
            best_val = self._results[-1][self._validation_k]["val_results"][self.validation_metric]
            self.best_metric_value = best_val
            self.logger.info(
                "Recorded best validation result",
                extra={"context": {"metric": self.validation_metric, "value": best_val, "iteration": it + 1}}
            )
            if self.config.meta.save_weights:
                if hasattr(self, "_model"):
                    writer.write_model(
                        obj=self._model.get_model_state(),
                        save_folder=self.global_config.path_output_rec_weight,
                        model_name=self.name
                    )
                else:
                    self.logger.warning("No model to save")

    def get_recs(self, k: int = 100):
        preds_test, preds_val = {}, {}
        dataloader = self._data.eval_dataloader(self.config.eval_batch_size)

        iter_data = tqdm(
            dataloader,
            desc="Collecting",
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

    def _compute_batch_recs(self, k, user_indices, item_indices=None):
        """Common logic for computing top-k recommendations."""
        if item_indices is not None:
            preds = self.model.predict_sampled(user_indices, item_indices)
            mask = item_indices == -1
        else:
            preds = self.model.predict_full(user_indices)
            eval_batch = self._data.sp_i_train_ratings[user_indices.tolist()]
            mask = eval_batch.nonzero()

        v, i = self._get_top_k(preds, k, mask, item_indices)
        recs_dict = self._get_recs_dict(v, i, user_indices)

        return recs_dict

    def _get_recs_dict(self, values, item_indices, user_indices):
        if not item_indices.size:
            return {}
        pr_users, pr_items = self._data.get_inverse_mappings()
        mapped_items = np.array(pr_items)[item_indices]
        mat = [[*zip(item, val)] for item, val in zip(mapped_items, values)]
        proc_batch = dict(zip([pr_users[u_i] for u_i in user_indices], mat))
        return proc_batch

    def _get_top_k(self, users_recs, k, mask, item_indices=None):
        users_recs[mask] = -torch.inf

        k = min(k, users_recs.shape[1])
        v, i = torch.topk(users_recs, k=k, sorted=True)

        if item_indices is not None:
            i = item_indices.gather(1, i)

        return v.numpy(), i.numpy()

    def restore_weights(self):
        try:
            weights = reader.read_model(
                read_folder=self.global_config.path_output_rec_weight,
                model_name=self.name
            )
            self.model.set_model_state(weights)
            self.evaluate()
            return True
        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

    def get_loss(self):
        if self.config.meta.optimize_internal_loss:
            return min(self._losses)
        else:
            return -max([r[self._validation_k]["val_results"][self.validation_metric] for r in self._results])

    def get_params(self):
        return self.config.model_dump(exclude={"meta"})

    def get_results(self):
        return self._results[self.get_best_arg()]

    def get_best_arg(self):
        if self.config.meta.optimize_internal_loss:
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
            total=int(self.model.transactions // self.config.batch_size),
            desc="Training",
            disable=not self.config.meta.verbose
        )
        with iter_ as t:
            for batch in dataloader:
                steps += 1
                loss += self.model.train_step(batch, *args)
                t.set_postfix({'loss': f'{loss / steps:.5f}'})
                t.update()
        return loss


class TraditionalTrainer(Trainer):
    def __init__(self, data, config, params, model_class):
        super().__init__(data, config, params, model_class)
        self.epochs = 1

    def _train_epoch(self, *args):
        self.model.initialize()
        return 0


class GeneralTrainer(AbstractTrainer):
    def __init__(self, data, config, params, model_class):
        super().__init__(data, config, params, model_class)
        self.optimizer = self.model.optimizer
        torch.manual_seed(self.config.seed)

    def _train_epoch(self, it, dataloader, *args):
        self.model.train()
        total_loss, steps = 0, 0
        iter_ = tqdm(
            total=int(self.model.transactions // self.config.batch_size),
            desc="Training",
            disable=not self.config.meta.verbose
        )
        with iter_ as t:
            for batch in dataloader:
                steps += 1
                self.optimizer.zero_grad()
                res = self.model.train_step(batch, steps, *args)
                loss, inputs = res if isinstance(res, tuple) else (res, None)
                loss.backward(inputs=inputs)
                total_loss += loss.detach().cpu().numpy()
                self.optimizer.step()
                t.set_postfix({'loss': f'{total_loss / steps:.5f}'})
                t.update()
        return total_loss

    @torch.no_grad()
    def evaluate(self, it=0, loss=0):
        self.model.eval()
        super().evaluate(it, loss)
