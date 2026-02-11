from typing import List, Dict, Optional

from elliot.namespace.common import BaseConfig


class EvaluationConfig(BaseConfig):
    """Evaluation configuration.

    Attributes:
        cutoffs (List[int]): List of cutoff values used for evaluation metrics. Defaults to [].
        simple_metrics (List[str]): List of simple evaluation metric names. Defaults to [].
        complex_metrics (Dict[str, dict]): Mapping of complex metrics to their parameters. Defaults to {}.
        relevance_threshold (int): Minimum relevance value to consider an interaction relevant. Defaults to 0.
        paired_ttest (Dict[str, dict]): Configuration for paired t-test comparisons. Defaults to {}.
        wilcoxon_test (Dict[str, dict]): Configuration for Wilcoxon signed-rank tests. Defaults to {}.
        user_filter_file (str, optional): Path to a file specifying users to include in evaluation.
    """

    cutoffs: List[int] = []
    simple_metrics: List[str] = []
    complex_metrics: Dict[str, dict] = {}
    relevance_threshold: int = 0
    paired_ttest: Dict[str, dict] = {}
    wilcoxon_test: Dict[str, dict] = {}
    user_filter_file: Optional[str] = None
