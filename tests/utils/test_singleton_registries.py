import pytest
from types import SimpleNamespace

from elliot.dataset.modular_loaders import AbstractLoader
from elliot.dataset.samplers import PipelineSampler
from elliot.evaluation.metrics import BaseMetric
from elliot.namespace import build_namespace
from elliot.recommender import BaseRecommender
from elliot.utils.registry import side_info_registry, sampler_registry, metric_registry, model_registry

from tests.params import config_path


_experiment_config = {
    "dataset": "demo",
    "data_config": {
        "strategy": "dataset",
        "dataset_path": "../data/{0}/dataset.tsv",
    }
}

def _sample_config(config_dict=None):
    config_dict = config_dict or _experiment_config
    return {"experiment": config_dict}


@side_info_registry.register(provides="", format="")
class CustomLoader(AbstractLoader):
    custom_param: int

    def __init__(self, users, items, ns):
        super().__init__(users, items, ns)
        self.custom_param = ns.custom_param

    def get_mapped(self):
        pass

    def filter(self, users, items):
        pass

    def create_namespace(self):
        pass


@sampler_registry.register()
class CustomSampler(PipelineSampler):
    def __init__(self, custom_param, **params):
        super().__init__(**params)
        self.custom_param = custom_param

    def sample(self, it):
        pass


@metric_registry.register()
class CustomMetric(BaseMetric):
    def __init__(
        self,
        recommendations,
        config,
        params,
        eval_objects
    ):
        super().__init__(recommendations, config, params, eval_objects)


@model_registry.register()
class CustomRecommender(BaseRecommender):
    custom_param: float

    def __init__(self, params, interactions, seed, *args, **kwargs):
        super().__init__(params, interactions, seed, *args, **kwargs)

    def train_step(self, batch, *args):
        pass

    def predict(self, user_indices, item_indices=None):
        pass

class _FakeInteractions:
    def __init__(self):
        self.dims = (0, 0)
        self.transactions = 0

    def get_users_items(self):
        return [], []


class TestSingletonRegistries:

    def test_side_info_registry(self):
        loader_name = CustomLoader.__name__

        config_data = _sample_config({
            **_experiment_config,
            "data_config": {
                **_experiment_config["data_config"],
                "side_information": {
                    "dataloader": loader_name,
                    "custom_param": 1
                }
            }
        })

        config = build_namespace(config_path=config_path, config_data=config_data)

        custom_loader_config = config.data_config.side_information[0]
        assert custom_loader_config.dataloader == loader_name
        assert custom_loader_config.custom_param == 1

        custom_loader = side_info_registry.get(
            name=loader_name,
            users=set(),
            items=set(),
            ns=custom_loader_config
        )
        assert custom_loader.custom_param == 1
        assert custom_loader.provides == ""
        assert custom_loader.format == ""

    def test_sampler_registry(self):
        sampler_name = CustomSampler.__name__

        custom_sampler = sampler_registry.get(
            name=sampler_name,
            train_dict={},
            transactions=0,
            users=[],
            items=[],
            n_users=0,
            n_items=0,
            seed=42,
            custom_param=1
        )

        assert custom_sampler.events == 0
        assert custom_sampler._users == []
        assert custom_sampler._items == []
        assert custom_sampler._nusers == 0
        assert custom_sampler._nitems == 0
        assert custom_sampler._indexed_ratings == {}
        assert custom_sampler.custom_param == 1

    def test_metric_registry(self):
        metric_name = CustomMetric.__name__

        config_data = _sample_config({
            **_experiment_config,
            "evaluation": {
                "simple_metrics": [metric_name],
            },
            "models": {
                "ItemKNN": {
                    "neighborhood": 50
                }
            }
        })

        config = build_namespace(config_path=config_path, config_data=config_data)

        assert metric_name in config.evaluation.simple_metrics

        custom_metric = metric_registry.get(
            name=metric_name,
            recommendations={},
            config=config,
            params=config.models["ItemKNN"],
            eval_objects=SimpleNamespace()
        )

        assert custom_metric.name == metric_name
        assert custom_metric._recommendations == {}
        assert custom_metric._config == config
        assert custom_metric._params == config.models["ItemKNN"]
        assert custom_metric._evaluation_objects == SimpleNamespace()
        assert custom_metric._additional_data is None

    def test_model_registry(self):
        model_name = CustomRecommender.__name__
        interactions = _FakeInteractions()

        config_data = _sample_config({
            **_experiment_config,
            "models": {
                model_name: {
                    "custom_param": 1.0
                }
            }
        })

        config = build_namespace(config_path=config_path, config_data=config_data)

        assert model_name in config.models
        recommender_config = config.models[model_name]
        assert recommender_config.custom_param == 1.0

        custom_recommender = model_registry.get(
            name=model_name,
            params=recommender_config,
            interactions=interactions,
            seed=42
        )

        assert custom_recommender.name == model_name
        assert custom_recommender._interactions == interactions
        assert custom_recommender._users == []
        assert custom_recommender._items == []
        assert custom_recommender._num_users == 0
        assert custom_recommender._num_items == 0
        assert custom_recommender._seed == 42
        assert "custom_param" in custom_recommender.params_list
        assert custom_recommender.custom_param == 1.0


class TestSingletonRegistriesFailures:

    def test_missing_side_info_registry(self):
        loader_name = "MissingLoader"

        config_data = _sample_config({
            **_experiment_config,
            "data_config": {
                **_experiment_config["data_config"],
                "side_information": {
                    "dataloader": loader_name,
                }
            }
        })

        config = build_namespace(config_path=config_path, config_data=config_data)

        assert config.data_config.side_information == []

        with pytest.raises(ValueError):
            side_info_registry.get(name=loader_name)

    def test_missing_sampler_registry(self):
        sampler_name = "MissingSampler"

        with pytest.raises(ValueError):
            sampler_registry.get(name=sampler_name)

    def test_missing_metric_registry(self):
        metric_name = "MissingMetric"

        config_data = _sample_config({
            **_experiment_config,
            "evaluation": {
                "simple_metrics": [metric_name]
            },
            "models": {
                "ItemKNN": {
                    "neighborhood": 50
                }
            }
        })

        config = build_namespace(config_path=config_path, config_data=config_data)

        assert metric_name not in config.evaluation.simple_metrics

        with pytest.raises(ValueError):
            metric_registry.get(
                name=metric_name,
                recommendations={},
                config=config,
                params=config.models["ItemKNN"],
                eval_objects=SimpleNamespace()
            )

    def test_missing_model_registry(self):
        model_name = "MissingModel"
        interactions = _FakeInteractions()

        config_data = _sample_config({
            **_experiment_config,
            "models": {
                model_name: {}
            }
        })

        config = build_namespace(config_path=config_path, config_data=config_data)

        assert model_name not in config.models

        with pytest.raises(ValueError):
            model_registry.get(
                name=model_name,
                params=None,
                interactions=interactions,
                seed=42
            )


if __name__ == '__main__':
    pytest.main()
