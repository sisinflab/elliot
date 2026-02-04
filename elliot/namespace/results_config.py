from typing import List
from pydantic import Field, field_validator

from elliot.namespace.read_write_config import WriterConfig
from elliot.namespace.common import BaseConfig


class ResultsConfig(BaseConfig):
    save_performance: bool = True
    save_performance_triplets: bool = False
    save_times: bool = True
    save_best_models: bool = True
    save_trials: bool = True
    trials_formats: List[str] = ["json", "tsv"]
    save_fold_stats: bool = True
    save_fold_stats_triplets: bool = False
    save_statistical: bool = False
    # writer: WriterConfig = Field(default_factory=WriterConfig)
