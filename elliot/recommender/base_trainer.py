import inspect
import random
import logging as pylog
import time

import torch
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.utils.data import DataLoader

from elliot.namespace import RecommenderConfig, ExperimentConfig
from elliot.dataset import DataSet
from elliot.evaluation.evaluator import Evaluator
from elliot.recommender import AbstractRecommender
from elliot.recommender.collector import get_recommendations
from elliot.recommender.early_stopping import EarlyStopping

from elliot.utils.read import Reader
from elliot.utils.write import Writer
from elliot.utils import logging, split_metric


class AbstractTrainer(ABC):
    model_config: RecommenderConfig
    model: AbstractRecommender

    def __init__(
        self,
        config: ExperimentConfig,
        model: AbstractRecommender,
        *args,
        **kwargs
    ):
        """
        This class represents a recommender model. You can load a pretrained model
        by specifying its checkpoint path and use it for training/testing purposes.

        Args:
            data: data loader object
            params: dictionary with all parameters
        """
        self.config = config
        self.model_config = model.model_config
        self.model = model

        # Logger
        package_name = inspect.getmodule(self.model.__class__).__package__
        rec_name = f"external.{self.model.name}" if "external" in package_name else self.model.name
        self.logger = logging.get_logger_model(
            rec_name,
            pylog.CRITICAL if self.config.config_test else pylog.DEBUG
        )
        self.model.logger = self.logger

        self.reader = Reader(self.logger)
        self.writer = Writer(self.logger)

        # Validation metric
        self._val_metric, self._val_k = split_metric(self.model_config.meta.validation_metric)

        if self.model_config.epochs < self.model_config.meta.validation_rate:
            raise ValueError(f"The first validation epoch ({self.model_config.meta.validation_rate}) "
                             f"is later than the overall number of epochs ({self.model_config.epochs}).")

        # Early stopping
        self._early_stopping = EarlyStopping(self.model_config.early_stopping)

        # Evaluator
        self.evaluator = Evaluator(self.config, self.model_config)

        # Set seed
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)

    @property
    def name(self):
        return self.model.name + f"_{self.get_base_params_shortcut()}" + self.model.name_param

    def get_base_params_shortcut(self):
        return "_".join([str(k) + "=" + str(v).replace(".", "$") for k, v in
                         dict({"seed": self.config.random_seed,
                               "epochs": self.model_config.epochs,
                               "batch_size": self.model_config.batch_size,
                               "eval_batch_size": self.model_config.eval_batch_size}).items()
                         ])

    # def get_model_params_shortcut(self):
    #     return "_".join(
    #         [str(p[2])+"="+ str(p[5](getattr(self._model, p[0]))
    #                             if p[5] else getattr(self._model, p[0])).replace(".", "$")
    #          for p in self._model.params_list]
    #     )

    def _init_train(self, disable_early_stopping=False):
        self._best_metric_value = 0
        self._losses = []
        self._val_results = []
        self._params_list = []
        self._epoch_train_history = []
        self._validation_history = []

        if disable_early_stopping:
            self._early_stopping.active = False

    def train(self, dataset, validate=True):
        self._init_train(disable_early_stopping=not validate)
        evaluation_set = "val"

        if self.model_config.meta.restore:
            return self.restore_weights(dataset, evaluation_set)

        self.logger.info(
            "Loaded training dataset",
            extra={"context": {"transactions": dataset.train_set.transactions}}
        )

        training_dataloader = self.model.get_training_dataloader(
            batch_size=self.model_config.batch_size
        )
        if not isinstance(training_dataloader, DataLoader):
            self.model_config.meta.verbose = False

        for it in self.iterate():
            self.logger.debug(
                "Starting iteration",
                extra={"context": {"iteration": it + 1, "epochs": self.model_config.epochs}}
            )

            start = time.perf_counter()
            loss = self._train_epoch(it, training_dataloader)
            end = time.perf_counter()

            try:
                train_loss = float(loss)
            except (TypeError, ValueError):
                train_loss = None
            if train_loss is not None and np.isfinite(train_loss):
                epoch_number = self._trend_epoch_for_iteration(it)
                self._epoch_train_history.append(
                    {"epoch": int(epoch_number), "train_loss": train_loss}
                )

            self.logger.debug(
                "Completed iteration",
                extra={"context": {"iteration": it + 1, "duration_sec": end - start}}
            )

            if not (it + 1) % self.model_config.meta.validation_rate and validate:
                self.logger.debug(f'Epoch {(it + 1)}/{self.model_config.epochs} loss {loss:.5f}')
                self._losses.append(loss)
                self.evaluate(dataset, it, evaluation_set)

        if validate:
            result_dict = self._val_results[self.get_best_arg()]
            return self.get_report(result_dict, evaluation_set)
        else:
            return {}

    def evaluate(self, dataset, it=None, evaluation_set="test"):
        if self.model_config.eval_batch_size is None:
            self.model_config.eval_batch_size = self.model_config.batch_size

        dataloader = dataset.get_eval_dataloader(
            batch_size=self.model_config.eval_batch_size,
            session_strategy=self.model_config.meta.session_strategy
        )
        k = self.evaluator.get_needed_recommendations()

        recs = get_recommendations(self.model, dataloader, dataset, k)
        result_dict = self.evaluator.eval(recs, dataset, evaluation_set)

        if self.model_config.meta.save_recs:
            self.logger.info(f"Writing recommendations at: {self.config.path_output_rec_result}")
            self.writer.write_recommendations(
                recommendations=recs[1],
                save_folder=self.config.path_output_rec_result,
                model_name=self.name,
                it=it,
                header=self.model_config.meta.rec_writer.header,
                columns=self.model_config.meta.rec_writer.columns,
                sep=self.model_config.meta.rec_writer.sep,
                ext=self.model_config.meta.rec_writer.ext
            )

        if it is not None:
            self._val_results.append(result_dict)

            epoch_number = int(self._trend_epoch_for_iteration(it))

            val_metric = result_dict[self._val_k]["val_results"][self._val_metric]
            if self._val_metric.upper() in {"MSE", "RMSE", "MAE"}:
                val_loss = float(val_metric)
            else:
                val_loss = 1.0 - float(val_metric)

            self._validation_history.append(
                {
                    "epoch": epoch_number,
                    "val_metric": float(val_metric),
                    "val_loss": val_loss,
                }
            )

            if (len(self._val_results) - 1) == self.get_best_arg():
                self.model_config.best_iteration = it + 1
                self._best_metric_value = val_metric

                self.logger.info(
                    "Recorded best validation result",
                    extra={"context": {"metric": self._val_metric, "value": val_metric, "iteration": it + 1}}
                )

                if self.model_config.meta.save_weights:
                    self.writer.write_model(
                        obj=self.model.get_model_state(),
                        save_folder=self.config.path_output_rec_weight,
                        model_name=self.name,
                        ext=self.model_config.meta.model_writer.ext
                    )

            return True

        return self.get_report(result_dict, evaluation_set)

    def restore_weights(self, dataset, evaluation_set):
        try:
            weights = self.reader.read_model(
                read_folder=self.config.path_output_rec_weight,
                model_name=self.name,
                ext=self.model_config.meta.model_reader.ext
            )
            self.model.set_model_state(weights)
            self.evaluate(dataset, evaluation_set=evaluation_set)
            return True
        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

    def get_loss(self):
        if self.model_config.meta.optimize_internal_loss:
            return min(self._losses)
        else:
            return -max([r[self._val_k]["val_results"][self._val_metric] for r in self._val_results])

    def get_params(self):
        return self.model_config.model_dump()

    def get_best_arg(self):
        if self.model_config.meta.optimize_internal_loss:
            val_results = np.argmin(self._losses)
        else:
            val_results = np.argmax(
                [r[self._val_k]["val_results"][self._val_metric] for r in self._val_results]
            )
        return val_results

    def get_report(self, results, evaluation_set="test"):
        return {
            "name": self.name,
            "params": self.get_params(),
            f"{evaluation_set}_results": {
                k: result_dict[f"{evaluation_set}_results"] for k, result_dict in results.items()
            },
            f"{evaluation_set}_statistical_results": {
                k: result_dict[f"{evaluation_set}_statistical_results"] for k, result_dict in results.items()
            }
        }

    def iterate(self):
        for iteration in range(self.model_config.epochs):
            stop, reasons = self._early_stopping.stop(self._losses[:], self._val_results)
            if stop:
                self.logger.info(f"Met Early Stopping conditions: {reasons}")
                break
            else:
                yield iteration

    def _trend_epoch_for_iteration(self, it: int) -> int:
        return int(it + 1)

    @abstractmethod
    def _train_epoch(self, it, dataloader, *args):
        raise NotImplementedError()

    #@staticmethod
    #def _batch_remove(original_str: str, char_list):
    #    for c in char_list:
    #        original_str = original_str.replace(c, "")
    #    return original_str


class BaseTrainer(AbstractTrainer):
    def __init__(self, config, model, *args, **kwargs):
        super().__init__(config, model, *args, **kwargs)

    def _train_epoch(self, it, dataloader, *args):
        total_loss = 0.0
        steps = 0
        iter_ = tqdm(
            total=int(self.model.transactions // self.model_config.batch_size),
            desc=f"Epoch {it + 1}/{self.model_config.epochs}",
            disable=not self.model_config.meta.verbose
        )
        with iter_ as t:
            for batch in dataloader:
                steps += 1
                batch_loss = self.model.train_step(batch, *args)
                if hasattr(batch_loss, "item"):
                    batch_loss = batch_loss.item()
                elif hasattr(batch_loss, "numpy"):
                    batch_loss = float(batch_loss)
                total_loss += batch_loss
                t.set_postfix({'loss': f'{total_loss / steps:.5f}'})
                t.update()
        return (total_loss / steps) if steps else 0.0


class TraditionalTrainer(BaseTrainer):
    def __init__(self, config, model, *args, **kwargs):
        super().__init__(config, model, *args, **kwargs)
        self.model_config.epochs = 1

    def _train_epoch(self, *args):
        self.model.initialize()
        return 0

    def _trend_epoch_for_iteration(self, it: int) -> int:
        return 0


class GeneralTrainer(AbstractTrainer):
    def __init__(self, config, model, *args, **kwargs):
        super().__init__(config, model, *args, **kwargs)
        self.optimizer = self.model.optimizer
        torch.manual_seed(self.config.random_seed)

    def _train_epoch(self, it, dataloader, *args):
        self.model.train()
        total_loss, steps = 0.0, 0
        iter_ = tqdm(
            total=int(self.model.transactions // self.model_config.batch_size),
            desc=f"Epoch {it + 1}/{self.model_config.epochs}",
            disable=not self.model_config.meta.verbose
        )
        with iter_ as t:
            for batch in dataloader:
                steps += 1
                self.optimizer.zero_grad()
                res = self.model.train_step(batch, steps, *args)
                loss, inputs = res if isinstance(res, tuple) else (res, None)
                loss.backward(inputs=inputs)
                total_loss += loss.detach().cpu().item()
                self.optimizer.step()
                t.set_postfix({'loss': f'{total_loss / steps:.5f}'})
                t.update()
        return (total_loss / steps) if steps else 0.0

    @torch.no_grad()
    def evaluate(self, dataset, it=None, evaluation_set="test"):
        self.model.eval()
        return super().evaluate(dataset, it, evaluation_set)
