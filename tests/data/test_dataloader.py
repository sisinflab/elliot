import pytest
import numpy as np

from elliot.dataset import DataSetLoader
from elliot.namespace import build_namespace
from elliot.utils import get_trainer, get_model
from elliot.utils.folder import path_joiner, check_path, parent_dir

from tests.params import params_neg_sampling_fail as p
from tests.utils import dataset_path

current_path = path_joiner(__file__)


def training_dataloader(config_dict):
    trainer = _load_data_and_get_trainer(config_dict)
    dataloader = trainer.model.get_training_dataloader(batch_size=trainer.model_config.batch_size)
    return dataloader

def eval_dataloader(config_dict):
    trainer = _load_data_and_get_trainer(config_dict)
    dataloader = trainer.data.eval_dataloader(batch_size=trainer.model_config.eval_batch_size)
    return dataloader

def _load_data_and_get_trainer(config_dict):
    config_data = {
        "experiment": {
            "dataset": "dataloader",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path,
                "reader": {"header": True}
            },
            "splitting": {
                "test_splitting": {
                    "strategy": "temporal_holdout",
                    "test_ratio": 0.1
                },
                "validation_splitting": {
                    "strategy": "temporal_holdout",
                    "test_ratio": 0.1
                }
            },
            "top_k": 10,
            "evaluation": {
                "simple_metrics": ["nDCG"]
            },
            **config_dict
        }
    }

    config = build_namespace(config_path=current_path, config_data=config_data)

    dataset_loader = DataSetLoader(config=config)
    data_list = dataset_loader.build()

    (model_name, model_config), = config.models.items()

    model_class = get_model(model_name, config)
    trainer_class = get_trainer(model_class)

    trainer = trainer_class(data_list[0][0], config, model_config, model_class)

    return trainer


class TestEvalDataloader:

    @pytest.mark.parametrize("leave_one_out", [False, True])
    def test_neg_random(self, leave_one_out):
        num_negatives = 20
        config = {
            "negative_sampling": {
                "strategy": "random",
                "num_negatives": num_negatives,
                "leave_one_out": leave_one_out
            },
            "models": {
                "ItemKNN": {
                    "meta": {"validation_metric": "nDCG"},
                    "eval_batch_size": 2
                }
            }
        }

        dataloader = eval_dataloader(config)

        positives = (
            np.array(range(0, 40)), np.array(0), np.array(range(0, 50))
        )
        expected_val_test_pos = (5, 1, 5) if not leave_one_out else (1, 1, 1)

        for batch in dataloader:
            for i, val, test in zip(*batch):
                i, val, test = i.numpy(), val.numpy(), test.numpy()
                val, test = val[val != -1], test[test != -1]
                if val.size and test.size:
                    num_val_pos = np.isin(val, positives[i]).sum()
                    num_test_pos = np.isin(test, positives[i]).sum()
                    assert num_val_pos == expected_val_test_pos[i]
                    assert num_test_pos == expected_val_test_pos[i]
                    assert len(val) - num_val_pos <= num_negatives
                    assert len(test) - num_test_pos <= num_negatives
                else:
                    assert i == 2

    @pytest.mark.parametrize("leave_one_out", [False, True])
    def test_neg_fixed(self, leave_one_out):
        config = {
            "negative_sampling": {
                "strategy": "fixed",
                "read_folder": "./{0}",
                "leave_one_out": leave_one_out
            },
            "models": {
                "ItemKNN": {
                    "meta": {"validation_metric": "nDCG"},
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
        expected_val_test_pos = (5, 1, 5) if not leave_one_out else (1, 1, 1)

        for batch in dataloader:
            for i, val, test in zip(*batch):
                i, val, test = i.numpy(), val.numpy(), test.numpy()
                val, test = val[val != -1], test[test != -1]
                if val.size and test.size:
                    num_val_pos = (~np.isin(val, val_neg[i])).sum()
                    num_test_pos = (~np.isin(test, test_neg[i])).sum()
                    assert num_val_pos == expected_val_test_pos[i]
                    assert num_test_pos == expected_val_test_pos[i]
                    assert np.isin(val_neg[i], val).all()
                    assert np.isin(test_neg[i], test).all()
                else:
                    assert i == 2

    def test_neg_saving_on_disk(self):
        save_folder = "./dataloader/negative"
        config = {
            "negative_sampling": {
                "strategy": "random",
                "num_negatives": 20,
                "save_on_disk": True,
                "save_folder": save_folder
            },
            "models": {
                "ItemKNN": {
                    "meta": {"validation_metric": "nDCG"},
                    "eval_batch_size": 2
                }
            }
        }

        eval_dataloader(config)

        train_path = path_joiner(parent_dir(current_path), save_folder, "val_negative.tsv")
        test_path = path_joiner(parent_dir(current_path), save_folder, "test_negative.tsv")
        assert check_path(train_path)
        assert check_path(test_path)


class TestEvalDataloaderFailures:

    @pytest.mark.parametrize("params", p["invalid_neg_random"])
    def test_invalid_params_neg_random(self, params):
        if params["num_negatives"] == 20 and params["leave_one_out"] == True:
            pytest.skip("Test requires at least one invalid parameter to be meaningful.")

        config = {
            "negative_sampling": {
                "strategy": "random",
                **params
            },
            "models": {
                "ItemKNN": {
                    "meta": {"validation_metric": "nDCG"},
                    "eval_batch_size": 2
                }
            }
        }

        with pytest.raises(ValueError):
            eval_dataloader(config)

    def test_invalid_save_folder_neg_random(self):
        config = {
            "negative_sampling": {
                "strategy": "random",
                "num_negatives": 10,
                "save_on_disk": True,
                "save_folder": 3
            },
            "models": {
                "ItemKNN": {
                    "meta": {"validation_metric": "nDCG"},
                    "eval_batch_size": 2
                }
            }
        }

        with pytest.raises(ValueError):
            eval_dataloader(config)

    @pytest.mark.parametrize("params", p["invalid_neg_fixed"])
    def test_invalid_or_missing_params_neg_fixed(self, params):
        if params["read_folder"] == "./{0}" and params["leave_one_out"] == True:
            pytest.skip("Test requires at least one invalid parameter to be meaningful.")

        config = {
            "negative_sampling": {
                "strategy": "fixed",
                **({"read_folder": params["read_folder"]} if params["read_folder"] is not None else {}),
                "leave_one_out": params["leave_one_out"]
            },
            "models": {
                "ItemKNN": {
                    "meta": {"validation_metric": "nDCG"},
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
                    "meta": {"validation_metric": "nDCG"},
                    "eval_batch_size": 2
                }
            }
        }

        with pytest.raises(ValueError):
            eval_dataloader(config)


if __name__ == "__main__":
    pytest.main()
