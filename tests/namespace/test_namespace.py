import pytest

from elliot.namespace import build_namespace
from elliot.utils.enums import (
    SearchSpace, DataLoadingStrategy, SplittingStrategy, NegativeSamplingStrategy, EvalTarget
)
from elliot.utils.folder import path_joiner, path_absolute, parent_dir

from tests.params import params_namespace_fail as p, config_path


def _sample_config(config_dict={}, load_as_session_only=False):
    config = {
        "experiment": {
            "dataset": "demo",
            "data_config": {
                "strategy": "dataset",
                **({"session_strategy": "session_only"} if load_as_session_only else {}),
                "dataset_path": "../data/{0}/dataset.tsv",
                "side_information": {
                    "dataloader": "ItemAttributes",
                    "attribute_file": "../data/{0}/map"
                }
            },
            "splitting": {
                "test_splitting": {
                    "strategy": "temporal_holdout",
                    **({"test_ratio": 0.2} if not load_as_session_only else {"leave_n_out": 1})
                }
            },
            "backend": "pytorch",
            "evaluation": {
                "simple_metrics": ["nDCG", "HR"],
                "protocol": {
                    "negative_sampling": {
                        "strategy": "random",
                        "num_negatives": 20
                    }
                }
            },
            "top_k": 10,
            "models": {
                "ItemKNN": {
                    "meta": {"validation_metric": "HR"},
                    "neighborhood": [50, 100],
                }
            },
            **config_dict
        }
    }
    return config

_flat_model = {"ItemKNN": {"meta": {"validation_metric": "HR"}, "neighborhood": 50}}
_session_model = {"SASRec": {"meta": {"session_strategy": "session_only"}}}


class TestNamespace:

    def _check_not_none(self, config, excluded=None):
        params = config.model_dump(exclude=excluded)
        for key, value in params.items():
            if config.model_fields[key].is_required():
                assert value is not None

    def test_resolve_paths_and_defaults(self):
        config_data = _sample_config({
            "models": {}
        })
        config = build_namespace(config_path=config_path, config_data=config_data)

        self._check_not_none(config)
        self._check_not_none(config.data_config)
        self._check_not_none(config.splitting)
        self._check_not_none(config.splitting.test_splitting)
        self._check_not_none(config.evaluation)
        self._check_not_none(config.evaluation.protocol.negative_sampling)

        expected_data_path = path_absolute(
            path_joiner(parent_dir(config_path), "..", "data", "demo")
        )

        side_info_config = config.data_config.side_information[0]

        assert side_info_config.attribute_file == path_joiner(expected_data_path, "map")
        assert config.data_config.dataset_path == path_joiner(expected_data_path, "dataset.tsv")
        assert config.data_config.sequential is False
        assert config.data_config.remove_duplicates is False
        assert config.splitting.save_folder == path_joiner(expected_data_path, "splitting")
        assert config.evaluation.protocol.negative_sampling.save_folder == expected_data_path
        assert config.path_output_rec_result.endswith(path_joiner("results", "demo", "recs"))
        assert config.path_output_rec_weight.endswith(path_joiner("results", "demo", "weights"))
        assert config.path_output_rec_performance.endswith(path_joiner("results", "demo", "performance"))

        assert config.data_config.strategy == DataLoadingStrategy.DATASET
        assert side_info_config.dataloader == "ItemAttributes"
        assert config.splitting.test_splitting.strategy == SplittingStrategy.TEMP_HOLDOUT
        assert config.splitting.test_splitting.test_ratio == 0.2
        assert config.evaluation.protocol.negative_sampling.strategy == NegativeSamplingStrategy.RANDOM
        assert config.evaluation.protocol.negative_sampling.num_negatives == 20
        assert config.backend == ["pytorch"]
        assert config.top_k == 10
        assert config.evaluation.simple_metrics == ["nDCG", "HR"]

    def test_parse_models(self):
        config = build_namespace(config_path=config_path, config_data=_sample_config())

        assert len(config.models) == 1
        (model_name, model_config), = config.models.items()
        assert model_name == "ItemKNN"

        self._check_not_none(model_config)
        self._check_not_none(model_config.meta)

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

    def test_validation_metric(self):
        config = build_namespace(config_path=config_path, config_data=_sample_config())
        (_, model_config), = config.models.items()

        assert config.results.default_metric == "nDCG@10"
        assert model_config.meta.validation_metric == "HR@10"

    def test_early_stopping_monitor(self):
        config_data = _sample_config({
            "models": {
                "BPRMF": {
                    "early_stopping": {"patience": 3}
                }
            }
        })
        config = build_namespace(config_path=config_path, config_data=config_data)
        (_, model_config), = config.models.items()

        assert config.results.default_metric == "nDCG@10"
        assert model_config.meta.validation_metric == "nDCG@10"
        assert model_config.early_stopping.monitor == "nDCG@10"

        # TODO: Put the following in the hyperoptimization tests
        # model_ns, space, max_evals, opt_alg = payload
        # assert hasattr(model_ns, "neighbors")
        # assert "neighbors" in space
        # assert max_evals >= 1
        # assert opt_alg is not None

    def test_evaluation_target(self):
        config_data = _sample_config({
            "splitting": {
                "test_splitting": {"strategy": "fixed_timestamp", "timestamp": 1000}
            },
            "evaluation": {
                "protocol": {"target": "successive"}
            },
            "models": _session_model
        }, load_as_session_only=True)

        config = build_namespace(config_path=config_path, config_data=config_data)

        assert config.evaluation.protocol.target == EvalTarget.SUCCESSIVE
        assert config.evaluation.protocol.mask_seen

    @pytest.mark.parametrize("models,expected", [
        (_flat_model, EvalTarget.ALL),
        (_session_model, EvalTarget.LAST),
        ({**_flat_model, **_session_model}, EvalTarget.LAST),
    ])
    def test_default_evaluation_target(self, models, expected):
        config_data = _sample_config({
            "models": models
        }, load_as_session_only=True)

        config = build_namespace(config_path=config_path, config_data=config_data)

        # A single target for the whole experiment, whatever each model's strategy
        assert config.evaluation.protocol.target == expected

    @pytest.mark.parametrize("target", [None, "last"])
    def test_mixed_session_strategies(self, target):
        config_data = _sample_config({
            "evaluation": {
                "simple_metrics": ["nDCG", "HR"],
                "protocol": {"target": target}
            },
            "models": {**_flat_model, **_session_model}
        }, load_as_session_only=True)

        config = build_namespace(config_path=config_path, config_data=config_data)

        assert config.mixes_session_strategies
        assert config.session_only_models == ["SASRec"]
        assert config.evaluation.protocol.target == EvalTarget.LAST

    def test_user_filter(self):
        config = build_namespace(config_path=config_path, config_data=_sample_config())

        # No filter by default: every user is evaluated
        assert config.evaluation.protocol.user_filter.path is None

        config_data = _sample_config({
            "evaluation": {
                "simple_metrics": ["nDCG", "HR"],
                "protocol": {
                    "user_filter": {"path": "users.tsv", "id_space": "private"}
                }
            }
        })
        config = build_namespace(config_path=config_path, config_data=config_data)

        # Resolved like every other path, relative to the config file
        expected_path = path_absolute(path_joiner(parent_dir(config_path), "users.tsv"))
        assert config.evaluation.protocol.user_filter.path == expected_path
        assert config.evaluation.protocol.user_filter.id_space == "private"


class TestNamespaceFailures:

    @pytest.mark.parametrize("params", p["invalid_target_for_mixed_session_strategies"])
    def test_invalid_target_for_mixed_session_strategies(self, params):
        config_data = _sample_config({
            "evaluation": {
                "protocol": {"target": params["target"]}
            },
            "models": {**_flat_model, **_session_model}
        })

        with pytest.raises(ValueError):
            build_namespace(config_path=config_path, config_data=config_data)

    @pytest.mark.parametrize("params", p["invalid_splitting_for_mixed_session_strategies"])
    def test_invalid_splitting_for_mixed_session_strategies(self, params):
        config_data = _sample_config({
            "splitting": {
                "test_splitting": params["test_splitting"]
            },
            "evaluation": {
                "protocol": {"target": "last"}
            },
            "models": {**_flat_model, **_session_model}
        })

        with pytest.raises(ValueError):
            build_namespace(config_path=config_path, config_data=config_data)

    @pytest.mark.parametrize("params", p["invalid_splitting_for_evaluation_target"])
    def test_invalid_evaluation_target(self, params):
        config_data = _sample_config({
            "splitting": params["splitting"],
            "evaluation": {
                "protocol": {"target": "last"}
            }
        })

        with pytest.raises(ValueError):
            build_namespace(config_path=config_path, config_data=config_data)

    @pytest.mark.parametrize("params", p["invalid_evaluation_target_for_session_models"])
    def test_invalid_evaluation_target_for_session_models(self, params):
        config_data = _sample_config({
            "evaluation": {
                "protocol": {"target": params["target"]}
            },
            "models": _session_model
        }, load_as_session_only=True)

        with pytest.raises(ValueError):
            build_namespace(config_path=config_path, config_data=config_data)


if __name__ == '__main__':
    pytest.main()
