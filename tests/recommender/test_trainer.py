import pytest
import logging
import torch

from elliot.namespace import RecommenderConfig
from elliot.recommender import base_trainer
from elliot.recommender.base_trainer import AbstractTrainer
from elliot.evaluation.protocol import CollectedRecs


class _StepModel(torch.nn.Module):
    """Model whose single weight records the last trained epoch."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def get_training_dataloader(self, batch_size):
        return [0]


class _FakeEvaluator:
    def __init__(self, values):
        self._values = iter(values)

    def get_needed_recommendations(self):
        return 10

    def eval(self, collected, dataset, evaluation_set):
        return {10: {"val_results": {"nDCG": next(self._values)}, "val_statistical_results": {}}}


class _FakeProtocol:
    def get_eval_dataloader(self, **kwargs):
        return None


class _FakeDataset:
    train_set = type("TrainSet", (), {"transactions": 1})
    protocol = _FakeProtocol()


class _EpochTrainer(AbstractTrainer):
    name = "epoch-trainer"

    def __init__(self, values):
        # Do not call AbstractTrainer.__init__ (too heavy for unit test)
        self.model_config = RecommenderConfig(batch_size=1, epochs=len(values))
        self.logger = logging.getLogger("epoch-trainer")
        self.model = _StepModel()
        self.evaluator = _FakeEvaluator(values)
        self._val_metric, self._val_k = "nDCG", 10

    def iterate(self):
        return range(self.model_config.epochs)

    def _train_epoch(self, it, dataloader, *args):
        with torch.no_grad():
            self.model.weight.fill_(it + 1)
        return 1.0


def test_train_restores_best_epoch_weights(monkeypatch):
    monkeypatch.setattr(
        base_trainer, "collect", lambda *args, **kwargs: CollectedRecs(recs={}, ground_truth={})
    )

    trainer = _EpochTrainer([0.1, 0.5, 0.2])
    trainer.train(_FakeDataset())

    # Best validation result at the 2nd epoch: its weights, not the 3rd's, are kept
    assert trainer.model.weight.item() == pytest.approx(2.0)
    assert trainer.model_config.best_iteration == 2


if __name__ == "__main__":
    pytest.main()
