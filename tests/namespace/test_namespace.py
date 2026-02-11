import pytest

from elliot.namespace import build_namespace
from elliot.utils.enums import SearchSpace, DataLoadingStrategy, SplittingStrategy, NegativeSamplingStrategy
from elliot.utils.folder import path_joiner, path_absolute, parent_dir

from tests.utils import test_path

config_path = path_joiner(test_path, "configs", "test.yml")


def _sample_config():
    return {
        "experiment": {
            "dataset": "demo",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": "../data/{0}/dataset.tsv",
                "side_information": {
                    "dataloader": "FeatureLoader1",
                    "folder_map_features": "../data/{0}/map"
                }
            },
            "splitting": {
                "test_splitting": {
                    "strategy": "temporal_holdout",
                    "test_ratio": 0.2,
                }
            },
            "negative_sampling": {
                "strategy": "random",
                "num_negatives": 20
            },
            "backend": "pytorch",
            "evaluation": {"simple_metrics": ["nDCG"]},
            "top_k": 10,
            "models": {
                "ItemKNN": {
                    "meta": {},
                    "neighborhood": [50, 100],
                }
            }
        }
    }


class TestNamespace:

    def _check_not_none(self, config, excluded=None):
        params = config.model_dump(exclude=excluded)
        for key, value in params.items():
            if config.model_fields[key].is_required():
                assert value is not None

    def test_resolve_paths_and_defaults(self):
        config_data = {
            **_sample_config(),
            "models": {}
        }
        config = build_namespace(config_path=config_path, config_data=config_data)

        self._check_not_none(config)
        self._check_not_none(config.data_config)
        self._check_not_none(config.splitting)
        self._check_not_none(config.splitting.test_splitting)
        self._check_not_none(config.negative_sampling)
        self._check_not_none(config.evaluation)

        expected_data_path = path_absolute(
            path_joiner(parent_dir(config_path), "..", "data", "demo")
        )
        side_info_config = config.data_config.side_information[0]
        assert hasattr(side_info_config, "folder_map_features")
        assert side_info_config.folder_map_features == path_joiner(expected_data_path, "map")
        assert config.data_config.dataset_path == path_joiner(expected_data_path, "dataset.tsv")
        assert config.splitting.save_folder == path_joiner(expected_data_path, "splitting")
        assert config.negative_sampling.save_folder == expected_data_path
        assert config.path_output_rec_result.endswith(path_joiner("results", "demo", "recs"))
        assert config.path_output_rec_weight.endswith(path_joiner("results", "demo", "weights"))
        assert config.path_output_rec_performance.endswith(path_joiner("results", "demo", "performance"))

        assert config.data_config.strategy == DataLoadingStrategy.DATASET
        assert config.data_config.side_information[0].dataloader == "FeatureLoader1"
        assert config.splitting.test_splitting.strategy == SplittingStrategy.TEMP_HOLDOUT
        assert config.splitting.test_splitting.test_ratio == 0.2
        assert config.negative_sampling.strategy == NegativeSamplingStrategy.RANDOM
        assert config.negative_sampling.num_negatives == 20
        assert config.backend == ["pytorch"]
        assert config.top_k == 10
        assert config.evaluation.simple_metrics == ["nDCG"]

    def test_parse_models(self):
        config = build_namespace(config_path=config_path, config_data=_sample_config())

        assert len(config.models) == 1
        (model_name, model_config), = config.models.items()
        assert model_name == "ItemKNN"

        self._check_not_none(model_config)
        self._check_not_none(model_config.meta, excluded={"validation_metric", "validation_k"})

        assert model_config.neighborhood == [50, 100]

    def test_prepare_fields_for_search(self):
        config = build_namespace(config_path=config_path, config_data=_sample_config())
        (_, model_config), = config.models.items()

        model_config.prepare_fields_for_search()

        fields = model_config.model_dump(exclude={"name", "best_iteration", "meta"})
        for value in fields.values():
            assert isinstance(value, list)
            assert len(value) >= 2
            assert value[0] == SearchSpace.CHOICE.value

        # TODO: Put the following in the hyperoptimization tests
        # model_ns, space, max_evals, opt_alg = payload
        # assert hasattr(model_ns, "neighbors")
        # assert "neighbors" in space
        # assert max_evals >= 1
        # assert opt_alg is not None


if __name__ == '__main__':
    pytest.main()
