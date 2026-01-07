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
