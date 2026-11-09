import pytest
from types import SimpleNamespace

from elliot.evaluation.evaluator import Evaluator
from elliot.evaluation.relevance.relevance import Relevance
from elliot.evaluation.protocol import CollectedRecs
from elliot.utils.logging import get_logger


def build_evaluator(row_owner_map=None):
    evaluator = Evaluator.__new__(Evaluator)
    evaluator._row_owner_map = row_owner_map
    evaluator._multi_row = row_owner_map is not None
    evaluator._paired_ttest = True
    evaluator._accelerate = False
    evaluator._metrics = ["Precision"]
    evaluator._complex_metrics = []
    evaluator._rel_threshold = 1
    evaluator._config = SimpleNamespace()
    evaluator._params = SimpleNamespace()
    evaluator.logger = get_logger("Evaluator")
    return evaluator

def build_initialized_evaluator():
    evaluator = build_evaluator()
    evaluator._k = [2]
    evaluator._initialized = True
    evaluator._data = SimpleNamespace(train_set=SimpleNamespace(dims=(2, 3)))
    evaluator._pop = None
    evaluator._pop_cache = None
    return evaluator


class TestEvaluator:

    def test_session_only_groups_and_averages_by_owner(self):
        evaluator = build_evaluator({"u1::s0": "u1", "u1::s1": "u1", "u2::s0": "u2"})

        aggregated = evaluator._aggregate_rows_to_users({
            "u1::s0": 1.0,
            "u1::s1": 0.0,
            "u2::s0": 0.5,
        })

        assert aggregated == {"u1": 0.5, "u2": 0.5}

    def test_session_only_empty_input_is_passed_through(self):
        evaluator = build_evaluator({})
        assert evaluator._aggregate_rows_to_users({}) == {}

    def test_session_only_metric_value_averaged_per_session_then_per_user(self):
        session_map = {"u1::s0": "u1", "u1::s1": "u1", "u2::s0": "u2"}
        evaluator = build_evaluator(session_map)

        eval_data = {
            "u1::s0": {"i3": 1},
            "u1::s1": {"i3": 1},
            "u2::s0": {"i3": 1},
        }
        recommendations = {
            "u1::s0": [("i3", 0.9), ("i1", 0.5)],  # hit -> precision@2 = 0.5
            "u1::s1": [("i1", 0.9), ("i2", 0.5)],  # miss -> precision@2 = 0.0
            "u2::s0": [("i3", 0.9), ("i1", 0.5)],  # hit -> precision@2 = 0.5
        }
        eval_obj = SimpleNamespace(cutoff=2, relevance=Relevance(eval_data, 1), additional_metrics=[])

        results, statistical_results = evaluator._process_eval_data(recommendations, eval_data, eval_obj, "test")

        # u1's two sessions (0.5, 0.0) average to 0.25; u2 has a single session at 0.5
        assert statistical_results["Precision"] == {"u1": pytest.approx(0.25), "u2": pytest.approx(0.5)}
        assert results["Precision"] == pytest.approx(0.375)


class TestEvalCollected:
    """`eval()` takes rows, ground truth and owners only from the collected recs."""

    recommendations = {
        "u1::0": [("i3", 0.9), ("i1", 0.5)],  # hit -> precision@2 = 0.5
        "u1::1": [("i1", 0.9), ("i2", 0.5)],  # miss -> precision@2 = 0.0
        "u2::0": [("i3", 0.9), ("i1", 0.5)],  # hit -> precision@2 = 0.5
    }
    ground_truth = {row: {"i3": 1.0} for row in recommendations}

    def test_multi_row_averages_rows_per_owner(self):
        evaluator = build_initialized_evaluator()
        collected = CollectedRecs(
            recs=self.recommendations,
            ground_truth=self.ground_truth,
            owner_map={"u1::0": "u1", "u1::1": "u1", "u2::0": "u2"}
        )

        result = evaluator.eval(collected, dataset=SimpleNamespace(), label="test")[2]

        assert result["test_statistical_results"]["Precision"] == {
            "u1": pytest.approx(0.25), "u2": pytest.approx(0.5)
        }
        assert result["test_results"]["Precision"] == pytest.approx(0.375)

    def test_single_row_averages_rows_directly(self):
        evaluator = build_initialized_evaluator()
        collected = CollectedRecs(recs=self.recommendations, ground_truth=self.ground_truth)

        result = evaluator.eval(collected, dataset=SimpleNamespace(), label="test")[2]

        # Without an owner map each row is its own user
        assert result["test_statistical_results"]["Precision"] == {
            "u1::0": pytest.approx(0.5), "u1::1": pytest.approx(0.0), "u2::0": pytest.approx(0.5)
        }
        assert result["test_results"]["Precision"] == pytest.approx(1 / 3)

    def test_empty_ground_truth_yields_no_results(self):
        evaluator = build_initialized_evaluator()
        collected = CollectedRecs(recs={}, ground_truth={})

        result = evaluator.eval(collected, dataset=SimpleNamespace(), label="test")[2]

        assert result == {"test_results": None, "test_statistical_results": None}


if __name__ == "__main__":
    pytest.main()
