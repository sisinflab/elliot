from typing import List
from pydantic import Field

from elliot.namespace.write_config import TabularWriterConfig
from elliot.namespace.common import BaseConfig


class ResultsConfig(BaseConfig):
    """Results configuration.

    Attributes:
        save_performance (bool): Whether to save performance metrics. Defaults to True.
        save_performance_triplets (bool): Whether to save performance data in triplet format.
            Defaults to False.
        save_times (bool): Whether to save execution time data. Defaults to True.
        save_best_models (bool): Whether to save the best performing models. Defaults to True.
        save_trials (bool): Whether to save trial data. Defaults to True.
        trials_formats (List[str]): Formats to save trial data. Defaults to ["json", "tabular"].
        save_fold_stats (bool): Whether to save fold statistics. Defaults to True.
        save_fold_stats_triplets (bool): Whether to save fold statistics in triplet format. Defaults to False.
        save_statistical (bool): Whether to save statistical analysis results. Defaults to False.
        writer (TabularWriterConfig): Writing configuration.
        default_metric (str): Default metric to use for evaluation (automatically set).
    """

    save_performance: bool = True
    save_performance_triplets: bool = False
    save_times: bool = True
    save_best_models: bool = True
    save_trials: bool = True
    trials_formats: List[str] = ["json", "tabular"]
    save_fold_stats: bool = True
    save_fold_stats_triplets: bool = False
    save_statistical: bool = False
    writer: TabularWriterConfig = Field(default_factory=TabularWriterConfig, exclude=True)
    default_metric: str = ""
