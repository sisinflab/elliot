"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from .common import check_range
from .config import ExperimentConfig, build_namespace
from .data_config import DataConfig, SideInformationConfig
from .evaluation_config import EvaluationConfig
from .models_config import RecommenderConfig, MetaConfig, EarlyStoppingConfig
from .negative_sampling_config import NegativeSamplingConfig
from .prefiltering_config import PreFilteringConfig
from .read_write_config import ReaderConfig, WriterConfig
from .splitting_config import SplittingConfig, SplittingSingleConfig
