import pytest

from elliot.namespace import NameSpaceModel
from elliot.utils.folder import path_joiner, path_absolute, check_dir

from tests.utils import test_path


def _sample_config():
    return {
        "experiment": {
            "dataset": "demo",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": "../data/{0}/dataset.tsv",
            },
            "models": {
                "ItemKNN": {
                    "meta": {},
                    "neighbors": [50, 100],
                }
            },
            "evaluation": {"simple_metrics": ["nDCG"]},
            "top_k": 10,
        }
    }


def test_fill_base_resolves_paths_and_defaults():
    base_elliot = path_joiner(test_path, "elliot_root")
    base_config = path_joiner(test_path, "configs")
    check_dir(base_elliot)
    check_dir(base_config)

    model = NameSpaceModel(_sample_config(), str(base_elliot), str(base_config))
    model.fill_base()
    ns = model.base_namespace

    expected_dataset = path_joiner(base_config, "../", "data", "demo", "dataset.tsv")
    assert ns.data_config.dataset_path == path_absolute(expected_dataset)
    assert ns.data_config.side_information == []
    assert ns.path_output_rec_result.endswith(path_joiner("results", "demo", "recs"))
    assert ns.path_output_rec_weight.endswith(path_joiner("results", "demo", "weights"))
    assert ns.path_output_rec_performance.endswith(path_joiner("results", "demo", "performance"))
    assert ns.backend == ["tensorflow"]
    assert ns.top_k == 10
    assert ns.evaluation.simple_metrics == ["nDCG"]


def test_fill_model_builds_hyperopt_space():
    base_elliot = path_joiner(test_path, "elliot_root")
    base_config = path_joiner(test_path, "configs")
    check_dir(base_elliot)
    check_dir(base_config)

    model = NameSpaceModel(_sample_config(), str(base_elliot), str(base_config))
    entries = list(model.fill_model())

    assert len(entries) == 1
    key, payload = entries[0]
    assert key == "ItemKNN"
    model_ns, space, max_evals, opt_alg = payload
    assert hasattr(model_ns, "neighbors")
    assert "neighbors" in space
    assert max_evals >= 1
    assert opt_alg is not None


if __name__ == '__main__':
    pytest.main()
