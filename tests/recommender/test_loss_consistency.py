import pytest
import logging
import torch

from types import SimpleNamespace

from elliot.namespace import RecommenderConfig
from elliot.recommender.base_trainer import AbstractTrainer, GeneralTrainer, BaseTrainer


class _DummyModel:
    def __init__(self, losses):
        self._losses = losses
        self.transactions = len(losses)

    def train_step(self, batch, *args):
        return self._losses[batch]


class _DummyTorchModel:
    def __init__(self, losses):
        self._losses = losses
        self.transactions = len(losses)

    def train(self):
        pass

    def train_step(self, batch, *args):
        return torch.tensor(self._losses[batch], requires_grad=True)


class _DummyOptimizer:
    def zero_grad(self):
        pass

    def step(self):
        pass


def _make_config(batch_size=1):
    return RecommenderConfig(
        batch_size=batch_size,
        epochs=1,
    )


def test_trainer_epoch_loss_is_mean():
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.model = _DummyModel([1.0, 2.0, 3.0])
    trainer.model_config = _make_config()

    loss = BaseTrainer._train_epoch(trainer, it=0, dataloader=[0, 1, 2])

    assert loss == pytest.approx(2.0)


def test_general_trainer_epoch_loss_is_mean():
    trainer = GeneralTrainer.__new__(GeneralTrainer)
    trainer.model = _DummyTorchModel([1.0, 3.0, 5.0])
    trainer.optimizer = _DummyOptimizer()
    trainer.model_config = _make_config()

    loss = GeneralTrainer._train_epoch(trainer, it=0, dataloader=[0, 1, 2])

    assert loss == pytest.approx(3.0)


class _DummyDataset:
    def __init__(self):
        self.train_set = SimpleNamespace(transactions=1)


class _DummyTrainer(AbstractTrainer):
    def __init__(self):
        # Do not call AbstractTrainer.__init__ (too heavy for unit test)
        self.model_config = _make_config()
        self.logger = logging.getLogger("dummy-trainer")
        self.model = SimpleNamespace(
            get_training_dataloader=lambda batch_size: [0],
        )

        self.last_loss = None

    def iterate(self):
        return range(self.model_config.epochs)

    def _train_epoch(self, it, dataloader, *args):
        return 5.0

    def evaluate(self, dataset, it=None, evaluation_set="test"):
        self.last_loss = self._losses[-1]
        self._val_results.append({})

    def get_report(self, results, evaluation_set="test"):
        return {}

    def get_best_arg(self):
        return 0


def test_train_passes_epoch_loss_to_evaluate():
    trainer = _DummyTrainer()
    dataset = _DummyDataset()
    trainer.train(dataset)

    assert trainer.last_loss == pytest.approx(5.0)


if __name__ == "__main__":
    pytest.main()
