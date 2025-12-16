import os
import importlib
from pathlib import Path
import pytest

path = Path(__file__).resolve().parent
namespace_model = getattr(importlib.import_module("elliot.namespace"), 'NameSpaceModel')


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


@pytest.mark.parametrize('tmp_path', [path])
def test_fill_base_resolves_paths_and_defaults(tmp_path):
    base_elliot = tmp_path / "elliot_root"
    base_config = tmp_path / "configs"
    base_elliot.mkdir(exist_ok=True)
    base_config.mkdir(exist_ok=True)

    model = namespace_model(_sample_config(), str(base_elliot), str(base_config))
    model.fill_base()
    ns = model.base_namespace

    expected_dataset = os.path.abspath(os.path.join(base_config, "../", "data", "demo", "dataset.tsv"))
    assert ns.data_config.dataset_path == expected_dataset
    assert ns.data_config.side_information == []
    assert ns.path_output_rec_result.endswith(os.path.join("results", "demo", "recs"))
    assert ns.path_output_rec_weight.endswith(os.path.join("results", "demo", "weights"))
    assert ns.path_output_rec_performance.endswith(os.path.join("results", "demo", "performance"))
    assert ns.backend == ["tensorflow"]
    assert ns.top_k == 10
    assert ns.evaluation.simple_metrics == ["nDCG"]


@pytest.mark.parametrize('tmp_path', [path])
def test_fill_model_builds_hyperopt_space(tmp_path):
    base_elliot = tmp_path / "elliot_root"
    base_config = tmp_path / "configs"
    base_elliot.mkdir(exist_ok=True)
    base_config.mkdir(exist_ok=True)

    model = namespace_model(_sample_config(), str(base_elliot), str(base_config))
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
