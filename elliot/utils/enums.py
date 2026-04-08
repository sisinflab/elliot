from enum import Enum


class DataLoadingStrategy(Enum):
    FIXED = "fixed"
    HIERARCHY = "hierarchy"
    DATASET = "dataset"


class PreFilteringStrategy(Enum):
    GLOBAL_TH = 'global_threshold'
    USER_AVG = 'user_average'
    USER_K_CORE = 'user_k_core'
    ITEM_K_CORE = 'item_k_core'
    ITER_K_CORE = 'iterative_k_core'
    N_ROUNDS_K_CORE = 'n_rounds_k_core'
    COLD_USERS = 'cold_users'


class SplittingStrategy(Enum):
    FIXED_TS = 'fixed_timestamp'
    TEMP_HOLDOUT = 'temporal_holdout'
    RAND_CV = 'random_cross_validation'
    RAND_SUB_SMP = 'random_subsampling'


class AlignmentMode(Enum):
    DROP = "drop"      # intersect with train (current behavior)
    PAD = "pad"        # add UNK/zero rows for missing users/items
    IMPUTE = "impute"  # fill missing with statistics/learned defaults


class Materialization(Enum):
    LAZY = "lazy"
    MEMORY = "memory"
    MMAP = "mmap"


class NegativeSamplingStrategy(Enum):
    RANDOM = 'random'
    FIXED = 'fixed'


class SamplerType(Enum):
    TRADITIONAL = 1
    PIPELINE = 2


class ModelType(Enum):
    BASE = 1
    TRADITIONAL = 2
    GENERAL = 3


class SearchSpace(Enum):
    CHOICE = "choice"
    RANDINT = "randint"
    UNIFORM = "uniform"
    QUNIFORM = "quniform"
    LOGUNIFORM = "loguniform"
    QLOGUNIFORM = "qloguniform"
    NORMAL = "normal"
    QNORMAL = "qnormal"
    LOGNORMAL = "lognormal"
    QLOGNORMAL = "qlognormal"


class OptimizationAlgorithm(Enum):
    TPE = "tpe"
    ATPE = "atpe"
    MIX = "mix"
    RAND = "rand"
    ANNEAL = "anneal"
    GRID = "grid"


class StatTest(Enum):
    PAIRED_TTEST = "paired_ttest"
    WILCOXON_TEST = "wilcoxon_test"
