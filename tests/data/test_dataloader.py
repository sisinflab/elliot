import pytest
import importlib
import numpy as np

from elliot.dataset import DataSetLoader
from elliot.recommender.utils import get_model
from elliot.utils.enums import DataLoadingStrategy, SplittingStrategy, NegativeSamplingStrategy
from elliot.utils.folder import parent_dir, path_joiner, check_path

from tests.params import params_neg_sampling_fail as p
from tests.utils import create_namespace, dataset_path

current_path = parent_dir(__file__)


def training_dataloader(config_dict):
    trainer = _load_data_and_get_model(config_dict)
    dataloader = trainer.model.get_training_dataloader(batch_size=trainer.config.batch_size)
    return dataloader

def eval_dataloader(config_dict):
    trainer = _load_data_and_get_model(config_dict)
    dataloader = trainer._data.eval_dataloader(batch_size=trainer.config.eval_batch_size)
    return dataloader

def _load_data_and_get_model(config_dict):
    config = {
        "experiment": {
            "dataset": "dataloader",
            "data_config": {
                "strategy": DataLoadingStrategy.DATASET.value,
                "dataset_path": dataset_path,
                "header": True,
                "columns": ["userId", "itemId", "", "timestamp"]
            },
            "splitting": {
                "test_splitting": {
                    "strategy": SplittingStrategy.TEMP_HOLDOUT.value,
                    "test_ratio": 0.1
                },
                "validation_splitting": {
                    "strategy": SplittingStrategy.TEMP_HOLDOUT.value,
                    "test_ratio": 0.1
                }
            },
            "top_k": 10,
            "evaluation": {
                "simple_metrics": ["nDCG", "HR"],
                "relevance_threshold": 0
            },
            **config_dict
        }
    }
    ns_model = create_namespace(config, current_path)
    ns = ns_model.base_namespace
    loader = DataSetLoader(ns)
    data_list = loader.build()

    models = [m for m in ns_model.fill_model()]
    key, params = models[0]
    model_class = getattr(importlib.import_module("elliot.recommender"), key)
    trainer = get_model(data_list[0][0], ns, params, model_class)
    return trainer


class TestEvalDataloader:

    def test_neg_random(self):
        config = {
            "negative_sampling": {
                "strategy": NegativeSamplingStrategy.RANDOM.value,
                "num_negatives": 20
            },
            "models": {
                "ItemKNN": {
                    "eval_batch_size": 2
                }
            }
        }

        dataloader = eval_dataloader(config)

        positives = (
            np.array(range(0, 40)), np.array(0), np.array(range(0, 50))
        )
        for batch in dataloader:
            for i, val, test in zip(*batch):
                i, val, test = i.numpy(), val.numpy(), test.numpy()
                if val.size and test.size:
                    assert np.isin(val, positives[i]).sum() == 1
                    assert np.isin(test, positives[i]).sum() == 1
                else:
                    assert i == 2

    def test_neg_fixed(self):
        config = {
            "negative_sampling": {
                "strategy": NegativeSamplingStrategy.FIXED.value,
                "read_folder": "./{0}"
            },
            "models": {
                "ItemKNN": {
                    "eval_batch_size": 2
                }
            }
        }

        dataloader = eval_dataloader(config)

        val_neg = (
            np.array(range(40, 45)), np.array(range(1, 6)), np.array([])
        )
        test_neg = (
            np.array(range(45, 50)), np.array(range(6, 11)), np.array([])
        )
        for batch in dataloader:
            for i, val, test in zip(*batch):
                i, val, test = i.numpy(), val.numpy(), test.numpy()
                if val.size and test.size:
                    assert np.isin(val_neg[i], val).all()
                    assert np.isin(test_neg[i], test).all()
                else:
                    assert i == 2

    def test_neg_saving_on_disk(self):
        save_folder = "./dataloader/negative"
        config = {
            "negative_sampling": {
                "strategy": NegativeSamplingStrategy.RANDOM.value,
                "num_negatives": 20,
                "save_on_disk": True,
                "save_folder": save_folder
            },
            "models": {
                "ItemKNN": {
                    "eval_batch_size": 2
                }
            }
        }

        eval_dataloader(config)

        train_path = path_joiner(current_path, save_folder, "val_negative.tsv")
        test_path = path_joiner(current_path, save_folder, "test_negative.tsv")
        assert check_path(train_path)
        assert check_path(test_path)


class TestEvalDataloaderFailures:

    @pytest.mark.parametrize("params", p["invalid_neg_random"])
    def test_invalid_params_neg_random(self, params):
        config = {
            "negative_sampling": {
                "strategy": NegativeSamplingStrategy.RANDOM.value,
                **params
            },
            "models": {
                "ItemKNN": {
                    "eval_batch_size": 2
                }
            }
        }

        with pytest.raises(ValueError):
            eval_dataloader(config)

    def test_invalid_save_folder_neg_random(self):
        config = {
            "negative_sampling": {
                "strategy": NegativeSamplingStrategy.RANDOM.value,
                "num_negatives": 10,
                "save_on_disk": True,
                "save_folder": 3
            },
            "models": {
                "ItemKNN": {
                    "eval_batch_size": 2
                }
            }
        }

        with pytest.raises(ValueError):
            eval_dataloader(config)

    @pytest.mark.parametrize("params", p["invalid_neg_fixed"])
    def test_invalid_or_missing_params_neg_fixed(self, params):
        config = {
            "negative_sampling": {
                "strategy": NegativeSamplingStrategy.FIXED.value,
                **({"read_folder": params["read_folder"]} if params["read_folder"] is not None else {}),
            },
            "models": {
                "ItemKNN": {
                    "eval_batch_size": 2
                }
            }
        }

        with pytest.raises((FileNotFoundError, ValueError, AttributeError)):
            eval_dataloader(config)

    @pytest.mark.parametrize("params", p["invalid_strategy"])
    def test_invalid_or_missing_strategy(self, params):
        config = {
            "negative_sampling": {
                **({"strategy": params["strategy"]} if params["strategy"] is not None else {}),
                "num_negatives": 10
            },
            "models": {
                "ItemKNN": {
                    "eval_batch_size": 2
                }
            }
        }

        with pytest.raises(ValueError):
            eval_dataloader(config)


if __name__ == "__main__":
    pytest.main()
