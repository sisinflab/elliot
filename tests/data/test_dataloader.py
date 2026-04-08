import pytest
import numpy as np

from elliot.dataset import DataSetLoader
from elliot.namespace import build_namespace
from elliot.utils.folder import path_joiner, check_path, parent_dir

from tests.params import params_neg_sampling_fail as p
from tests.utils import dataset_path

current_path = path_joiner(__file__)


def training_dataloader(config_dict, sampler_name, batch_size, **kwargs):
    val_data, _ = _load_data(config_dict)
    dataloader = val_data.train_set.get_dataloader(sampler_name, batch_size=batch_size, **kwargs)
    return dataloader

def eval_dataloader(config_dict, batch_size):
    val_data, main_data = _load_data(config_dict)
    val_dataloader = val_data.get_eval_dataloader(batch_size=batch_size)
    test_dataloader = main_data.get_eval_dataloader(batch_size=batch_size)
    return val_dataloader, test_dataloader

def _load_data(config_dict):
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

    loader = DataSetLoader(config=config)
    val_data, main_data = loader.build()

    loader.prepare_dataset(val_data, main_data)

    return val_data[0][0], main_data[0]


class TestEvalDataloader:

    @pytest.mark.parametrize("leave_one_out", [False, True])
    def test_neg_random(self, leave_one_out):
        num_negatives = 20
        config = {
            "negative_sampling": {
                "strategy": "random",
                "num_negatives": num_negatives,
                "leave_one_out": leave_one_out
            }
        }

        val_dataloader, test_dataloader = eval_dataloader(config, batch_size=2)

        train_val_positives = (
            np.array(range(0, 40)), np.array(0), np.array(list(range(0, 5)) + list(range(10, 50)))
        )
        train_test_positives = (
            np.array(range(0, 40)), np.array(0), np.array(range(5, 50))
        )

        expected_eval_pos = (5, 1, 5) if not leave_one_out else (1, 1, 1)

        def check_dataloader(dataloader, positives):
            for batch in dataloader:
                for i, eval_ in zip(*batch):
                    i, eval_ = i.numpy(), eval_.numpy()
                    eval_ = eval_[eval_ != -1]
                    num_eval_pos = np.isin(eval_, positives[i]).sum()
                    assert num_eval_pos == expected_eval_pos[i]
                    assert len(eval_) - num_eval_pos <= num_negatives

        check_dataloader(val_dataloader, train_val_positives)
        check_dataloader(test_dataloader, train_test_positives)

    @pytest.mark.parametrize("leave_one_out", [False, True])
    def test_neg_fixed(self, leave_one_out):
        config = {
            "negative_sampling": {
                "strategy": "fixed",
                "read_folder": "./{0}",
                "leave_one_out": leave_one_out
            }
        }

        val_dataloader, test_dataloader = eval_dataloader(config, batch_size=2)

        val_negatives = (
            np.array(range(40, 45)), np.array(range(1, 6)), np.array([])
        )
        test_negatives = (
            np.array(range(45, 50)), np.array(range(6, 11)), np.array([])
        )
        expected_val_test_pos = (5, 1, 5) if not leave_one_out else (1, 1, 1)

        def check_dataloader(dataloader, negatives):
            for batch in dataloader:
                for i, eval_ in zip(*batch):
                    i, eval_ = i.numpy(), eval_.numpy()
                    eval_ = eval_[eval_ != -1]
                    if eval_.size:
                        num_eval_pos = (~np.isin(eval_, negatives[i])).sum()
                        assert num_eval_pos == expected_val_test_pos[i]
                        assert np.isin(negatives[i], eval_).all()
                    else:
                        assert i == 2

        check_dataloader(val_dataloader, val_negatives)
        check_dataloader(test_dataloader, test_negatives)

    def test_neg_saving_on_disk(self):
        save_folder = "./dataloader/negative"
        config = {
            "negative_sampling": {
                "strategy": "random",
                "num_negatives": 20,
                "save_on_disk": True,
                "save_folder": save_folder
            }
        }

        eval_dataloader(config, batch_size=2)

        val_path = path_joiner(parent_dir(current_path), save_folder, "test1_val1_negative.tsv")
        test_path = path_joiner(parent_dir(current_path), save_folder, "test1_negative.tsv")
        assert check_path(val_path)
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
            }
        }

        with pytest.raises(ValueError):
            eval_dataloader(config, batch_size=2)

    def test_invalid_save_folder_neg_random(self):
        config = {
            "negative_sampling": {
                "strategy": "random",
                "num_negatives": 10,
                "save_on_disk": True,
                "save_folder": 3
            }
        }

        with pytest.raises(ValueError):
            eval_dataloader(config, batch_size=2)

    @pytest.mark.parametrize("params", p["invalid_neg_fixed"])
    def test_invalid_or_missing_params_neg_fixed(self, params):
        if params["read_folder"] == "./{0}" and params["leave_one_out"] == True:
            pytest.skip("Test requires at least one invalid parameter to be meaningful.")

        config = {
            "negative_sampling": {
                "strategy": "fixed",
                **({"read_folder": params["read_folder"]} if params["read_folder"] is not None else {}),
                "leave_one_out": params["leave_one_out"]
            }
        }

        with pytest.raises((FileNotFoundError, ValueError, AttributeError)):
            eval_dataloader(config, batch_size=2)

    @pytest.mark.parametrize("params", p["invalid_strategy"])
    def test_invalid_or_missing_strategy(self, params):
        config = {
            "negative_sampling": {
                **({"strategy": params["strategy"]} if params["strategy"] is not None else {}),
                "num_negatives": 10
            }
        }

        with pytest.raises(ValueError):
            eval_dataloader(config, batch_size=2)


if __name__ == "__main__":
    pytest.main()
