"""
Evaluation policies for hyperparameter search vs final evaluation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationPolicy:
    include_test: bool


class SearchPolicy(EvaluationPolicy):
    def __init__(self):
        super().__init__(include_test=False)


class FinalPolicy(EvaluationPolicy):
    def __init__(self):
        super().__init__(include_test=True)
