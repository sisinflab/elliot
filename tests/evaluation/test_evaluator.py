import pytest
from types import SimpleNamespace

from elliot.evaluation.evaluator import Evaluator
from elliot.evaluation.relevance.relevance import Relevance
from elliot.utils.logging import get_logger


def build_evaluator(session_owner_map=None):
    evaluator = Evaluator.__new__(Evaluator)
    evaluator._session_owner_map = session_owner_map
    evaluator._session_only = session_owner_map is not None
    evaluator._paired_ttest = True
    evaluator._accelerate = False
    evaluator._metrics = ["Precision"]
    evaluator._complex_metrics = []
    evaluator._rel_threshold = 1
    evaluator._config = SimpleNamespace()
    evaluator._params = SimpleNamespace()
    evaluator.logger = get_logger("Evaluator")
    return evaluator


class TestEvaluator:

    def test_session_only_groups_and_averages_by_owner(self):
        evaluator = build_evaluator({"u1::s0": "u1", "u1::s1": "u1", "u2::s0": "u2"})

        aggregated = evaluator._aggregate_sessions_to_users({
            "u1::s0": 1.0,
            "u1::s1": 0.0,
            "u2::s0": 0.5,
        })

        assert aggregated == {"u1": 0.5, "u2": 0.5}

    def test_session_only_empty_input_is_passed_through(self):
        evaluator = build_evaluator({})
        assert evaluator._aggregate_sessions_to_users({}) == {}

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


if __name__ == "__main__":
    pytest.main()
