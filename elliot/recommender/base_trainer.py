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
from elliot.recommender.early_stopping import EarlyStopping

from elliot.utils.read import Reader
from elliot.utils.write import Writer
from elliot.utils import logging, split_metric

reader = Reader()
writer = Writer()


class AbstractTrainer(ABC):
    model_config: RecommenderConfig

    def __init__(
        self,
        data: DataSet,
        config: ExperimentConfig,
        model_config: RecommenderConfig,
        model_class,
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
        self.data = data
        self.config = config
        self.model_config = model_config

        # Logger
        package_name = inspect.getmodule(model_class).__package__
        rec_name = f"external.{model_class.__name__}" if "external" in package_name else model_class.__name__
        self.logger = logging.get_logger_model(
            rec_name,
            pylog.CRITICAL if self.config.config_test else pylog.DEBUG
        )

        # Validation metric
        self._val_metric, self._val_k = split_metric(self.model_config.meta.validation_metric)

        # Early stopping
        self._early_stopping = EarlyStopping(self.model_config.early_stopping)

        if self.model_config.epochs < self.model_config.meta.validation_rate:
            raise ValueError(f"The first validation epoch ({self.model_config.meta.validation_rate}) "
                             f"is later than the overall number of epochs ({self.model_config.epochs}).")

        if self.model_config.eval_batch_size is None:
            self.model_config.eval_batch_size = self.model_config.batch_size

        # Model
        model_cfg = self.model_config.model_copy(deep=True)
        self.model = model_class(data, model_cfg, self.config.random_seed, self.logger)
        self.model_config.name = self.model.name

        # Set seed
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)

        # Further parameters
        self._num_items = data.num_items
        self._num_users = data.num_users

        self.best_metric_value = 0

        self._losses = []
        self._results = []
        self._params_list = []

        # Evaluator
        self.evaluator = Evaluator(data, model_config)

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

    def train(self):
        if self.model_config.meta.restore:
            return self.restore_weights()

        self.logger.info(
            "Loaded training dataset",
            extra={"context": {"transactions": self.data.transactions}}
        )
        training_dataloader = self.model.get_training_dataloader(self.model_config.batch_size)

        if not isinstance(training_dataloader, DataLoader):
            self.model_config.meta.verbose = False

        for it in self.iterate(self.model_config.epochs):
            self.logger.debug(
                "Starting iteration",
                extra={"context": {"iteration": it + 1, "epochs": self.model_config.epochs}}
            )
            start = time.perf_counter()
            loss = self._train_epoch(it, training_dataloader)
            end = time.perf_counter()
            self.logger.debug(
                "Completed iteration",
                extra={"context": {"iteration": it + 1, "duration_sec": end - start}}
            )
            if not (it + 1) % self.model_config.meta.validation_rate:
                self.evaluate(it, loss)

        return self.get_report()

    def evaluate(self, it=0, loss=0):
        recs = self.get_recs(self.evaluator.get_needed_recommendations())
        result_dict = self.evaluator.eval(recs)

        self._losses.append(loss)

        self._results.append(result_dict)

        # if it is not None:
        self.logger.debug(f'Epoch {(it + 1)}/{self.model_config.epochs} loss {loss:.5f}')
        # else:
        #    self.logger.info(f'Finished')

        if self.model_config.meta.save_recs:
            self.logger.info(f"Writing recommendations at: {self.config.path_output_rec_result}")
            # if it is not None:
            writer.write_recommendations(
                recommendations=recs[1],
                save_folder=self.config.path_output_rec_result,
                model_name=self.name,
                it=it,
                header=self.model_config.meta.rec_writer.header,
                columns=self.model_config.meta.rec_writer.columns,
                sep=self.model_config.meta.rec_writer.sep,
                ext=self.model_config.meta.rec_writer.ext
            )
            # else:
            #    store_recommendation(recs[1], os.path.abspath(
            #        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

        if (len(self._results) - 1) == self.get_best_arg():
            # if it is not None:
            self.config.best_iteration = it + 1
            best_val = self._results[-1][self._val_k]["val_results"][self._val_metric]
            self.best_metric_value = best_val
            self.logger.info(
                "Recorded best validation result",
                extra={"context": {"metric": self._val_metric, "value": best_val, "iteration": it + 1}}
            )
            if self.model_config.meta.save_weights:
                writer.write_model(
                    obj=self.model.get_model_state(),
                    save_folder=self.config.path_output_rec_weight,
                    model_name=self.name,
                    ext=self.model_config.meta.model_writer.ext
                )

    def get_recs(self, k: int = 100):
        preds_test, preds_val = {}, {}
        dataloader = self.data.eval_dataloader(self.model_config.eval_batch_size)

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
            eval_batch = self.data.sp_i_train_ratings[user_indices.tolist()]
            mask = eval_batch.nonzero()

        v, i = self._get_top_k(preds, k, mask, item_indices)
        recs_dict = self._get_recs_dict(v, i, user_indices)

        return recs_dict

    def _get_recs_dict(self, values, item_indices, user_indices):
        if not item_indices.size:
            return {}
        pr_users, pr_items = self.data.get_inverse_mappings()
        mapped_items = np.array(pr_items)[item_indices]
        mat = [[*zip(item, val)] for item, val in zip(mapped_items, values)]
        proc_batch = dict(zip([pr_users[u_i] for u_i in user_indices], mat))
        return proc_batch

    def _get_top_k(self, users_recs, k, mask, item_indices=None):
        device = users_recs.device
        if item_indices is not None and item_indices.device != device:
            item_indices = item_indices.to(device)
        # if isinstance(mask, tuple):
        #     mask = (
        #         torch.as_tensor(mask[0], device=device),
        #         torch.as_tensor(mask[1], device=device),
        #     )
        if isinstance(mask, np.ndarray):
            mask = torch.as_tensor(mask, device=device)
        elif isinstance(mask, torch.Tensor) and mask.device != device:
            mask = mask.to(device)

        users_recs[mask] = -torch.inf

        k = min(k, users_recs.shape[1])
        v, i = torch.topk(users_recs, k=k, sorted=True)

        if item_indices is not None:
            i = item_indices.gather(1, i)

        return v.detach().cpu().numpy(), i.detach().cpu().numpy()

    def restore_weights(self):
        try:
            weights = reader.read_model(
                read_folder=self.config.path_output_rec_weight,
                model_name=self.name,
                ext=self.model_config.meta.model_reader.ext
            )
            self.model.set_model_state(weights)
            self.evaluate()
            return True
        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

    def get_loss(self):
        if self.model_config.meta.optimize_internal_loss:
            return min(self._losses)
        else:
            return -max([r[self._val_k]["val_results"][self._val_metric] for r in self._results])

    def get_params(self):
        return self.model_config.model_dump()

    def get_results(self):
        return self._results[self.get_best_arg()]

    def get_best_arg(self):
        if self.model_config.meta.optimize_internal_loss:
            val_results = np.argmin(self._losses)
        else:
            val_results = np.argmax(
                [r[self._val_k]["val_results"][self._val_metric] for r in self._results])
        return val_results

    def get_report(self):
        results = self.get_results()
        return {
            "name": self.name,
            "params": self.get_params(),
            "val_results": {
                k: result_dict["val_results"] for k, result_dict in results.items()
            },
            "val_statistical_results": {
                k: result_dict["val_statistical_results"] for k, result_dict in results.items()
            },
            "test_results": {
                k: result_dict["test_results"] for k, result_dict in results.items()
            },
            "test_statistical_results": {
                k: result_dict["test_statistical_results"] for k, result_dict in results.items()
            }
        }

    def iterate(self, epochs):
        for iteration in range(epochs):
            stop, reasons = self._early_stopping.stop(self._losses[:], self._results)
            if stop:
                self.logger.info(f"Met Early Stopping conditions: {reasons}")
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


class TraditionalTrainer(Trainer):
    def __init__(self, data, config, params, model_class):
        super().__init__(data, config, params, model_class)
        self.model_config.epochs = 1

    def _train_epoch(self, *args):
        self.model.initialize()
        return 0


class GeneralTrainer(AbstractTrainer):
    def __init__(self, data, config, params, model_class):
        super().__init__(data, config, params, model_class)
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
    def evaluate(self, it=0, loss=0):
        self.model.eval()
        super().evaluate(it, loss)
