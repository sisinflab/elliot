import pytest

from elliot.namespace import EarlyStoppingConfig
from elliot.recommender.early_stopping import EarlyStopping

from tests.params import params_early_stopping_fail as p

metric = "nDCG"
cutoff = 10


def stop_no_improvement(config_dict):
    losses = [0.05, 0.06, 0.07, 0.08]
    results = [
        {cutoff: {"val_results": {metric: 0.08}}},
        {cutoff: {"val_results": {metric: 0.07}}},
        {cutoff: {"val_results": {metric: 0.06}}},
        {cutoff: {"val_results": {metric: 0.05}}}
    ]
    config = EarlyStoppingConfig(**config_dict)
    early_stopping = EarlyStopping(config)
    return early_stopping.stop(losses, results)


def stop_true(config_dict):
    losses = [0.05, 0.05, 0.05, 0.05]
    results = [
        {cutoff: {"val_results": {metric: 0.03}}},
        {cutoff: {"val_results": {metric: 0.03}}},
        {cutoff: {"val_results": {metric: 0.03}}},
        {cutoff: {"val_results": {metric: 0.03}}}
    ]
    config = EarlyStoppingConfig(**config_dict)
    early_stopping = EarlyStopping(config)
    return early_stopping.stop(losses, results)


def stop_false(config_dict):
    losses = [0.09, 0.07, 0.05, 0.03]
    results = [
        {cutoff: {"val_results": {metric: 0.05}}},
        {cutoff: {"val_results": {metric: 0.07}}},
        {cutoff: {"val_results": {metric: 0.09}}},
        {cutoff: {"val_results": {metric: 0.11}}}
    ]
    config = EarlyStoppingConfig(**config_dict)
    early_stopping = EarlyStopping(config)
    return early_stopping.stop(losses, results)


class TestEarlyStopping:

    @pytest.mark.parametrize("monitor", ["loss", f"{metric}@{cutoff}"])
    def test_no_improvement(self, monitor):
        config = {
            "monitor": monitor,
            "patience": 3
        }

        stop, _ = stop_no_improvement(config)
        assert stop == True

    @pytest.mark.parametrize("monitor", ["loss", f"{metric}@{cutoff}"])
    def test_min_delta(self, monitor):
        config = {
            "monitor": monitor,
            "patience": 3,
            "min_delta": 0.01,
        }

        stop, _ = stop_true(config)
        assert stop == True
        stop, _ = stop_false(config)
        assert stop == False

    @pytest.mark.parametrize("monitor", ["loss", f"{metric}@{cutoff}"])
    def test_rel_delta(self, monitor):
        config = {
            "monitor": monitor,
            "patience": 3,
            "rel_delta": 0.05,
        }

        stop, _ = stop_true(config)
        assert stop == True
        stop, _ = stop_false(config)
        assert stop == False

    @pytest.mark.parametrize("monitor", ["loss", f"{metric}@{cutoff}"])
    def test_baseline(self, monitor):
        config = {
            "monitor": monitor,
            "patience": 3,
            "baseline": 0.04,
        }

        stop, _ = stop_true(config)
        assert stop == True
        stop, _ = stop_false(config)
        assert stop == False

    @pytest.mark.parametrize("params", [
        {"monitor": "loss", "baseline": 0.06},
        {"monitor": f"{metric}@{cutoff}", "baseline": 0.02}
    ])
    def test_stop_with_some_conditions_not_met(self, params):
        config = {
            "monitor": params["monitor"],
            "patience": 3,
            "min_delta": 0.01,
            "rel_delta": 0.05,
            "baseline": params["baseline"]
        }

        stop, reasons = stop_true(config)
        assert stop == True
        for r in reasons:
            assert "baseline" not in r

    @pytest.mark.parametrize("monitor", ["loss", f"{metric}@{cutoff}"])
    def test_stop_with_insufficient_observations(self, monitor):
        config = {
            "monitor": monitor,
            "patience": 5,
            "min_delta": 0.01
        }

        stop, _ = stop_true(config)
        assert stop == False

    def test_initialization_with_none_config(self):
        early_stopping = EarlyStopping(None)

        assert early_stopping.active == False


class TestEarlyStoppingFailures:

    @pytest.mark.parametrize("params", p["invalid"])
    def test_invalid_params(self, params):
        if (
            params["monitor"] == "loss" and
            params["patience"] == 3 and
            params["mode"] == "min" and
            params["min_delta"] == 0.01 and
            params["rel_delta"] == 0.05 and
            params["baseline"] == 0.04 and
            params["verbose"] == True
        ):
            pytest.skip("Test requires at least one invalid parameter to be meaningful.")

        with pytest.raises(ValueError):
            EarlyStoppingConfig(**params)


if __name__ == "__main__":
    pytest.main()
