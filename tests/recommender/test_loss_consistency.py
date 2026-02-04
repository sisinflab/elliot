import pytest
import logging
import torch

from types import SimpleNamespace

from elliot.namespace import RecommenderConfig
from elliot.recommender.base_trainer import AbstractTrainer, GeneralTrainer, Trainer


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
    trainer = Trainer.__new__(Trainer)
    trainer.model = _DummyModel([1.0, 2.0, 3.0])
    trainer.model_config = _make_config()

    loss = Trainer._train_epoch(trainer, it=0, dataloader=[0, 1, 2])

    assert loss == pytest.approx(2.0)


def test_general_trainer_epoch_loss_is_mean():
    trainer = GeneralTrainer.__new__(GeneralTrainer)
    trainer.model = _DummyTorchModel([1.0, 3.0, 5.0])
    trainer.optimizer = _DummyOptimizer()
    trainer.model_config = _make_config()

    loss = GeneralTrainer._train_epoch(trainer, it=0, dataloader=[0, 1, 2])

    assert loss == pytest.approx(3.0)


class _DummyTrainer(AbstractTrainer):
    def __init__(self):
        # Do not call AbstractTrainer.__init__ (too heavy for unit test)
        self.data = SimpleNamespace(transactions=1)
        self.model_config = _make_config()
        self.logger = logging.getLogger("dummy-trainer")
        self.model = SimpleNamespace(
            transactions=1,
            get_training_dataloader=lambda batch_size: [0],
        )
        self.last_loss = None

    def iterate(self, epochs):
        return range(epochs)

    def _train_epoch(self, it, dataloader, *args):
        return 5.0

    def evaluate(self, it=0, loss=0):
        self.last_loss = loss

    def get_report(self):
        return {}


def test_train_passes_epoch_loss_to_evaluate():
    trainer = _DummyTrainer()
    trainer.train()

    assert trainer.last_loss == pytest.approx(5.0)


if __name__ == "__main__":
    pytest.main()
