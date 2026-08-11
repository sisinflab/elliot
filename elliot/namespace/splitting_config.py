from typing import Any, Optional
from pydantic import Field, model_validator

from elliot.namespace.common import BaseConfig
from elliot.namespace.write_config import TabularWriterConfig, SequenceWriterConfig
from elliot.utils.enums import SplittingStrategy
from elliot.utils.folder import path_joiner


class SplittingSingleConfig(BaseConfig):
    """Splitting configuration.

    Attributes:
        strategy (SplittingStrategy): Splitting strategy to apply.
        timestamp (float, optional): Optional timestamp for splitting. Defaults to None.
        min_below (int): Minimum number of items below threshold. Defaults to 1, min is 1.
        min_over (int): Minimum number of items over threshold. Defaults to 1, min is 1.
        test_ratio (float, optional): Fraction of data for testing; min is 0.1, max is 0.9.
        leave_n_out (int, optional): Number of items to leave out for test. Defaults to None.
        folds (int): Number of folds for cross-validation. Defaults to 5, min is 1, max is 20.
    """

    strategy: SplittingStrategy
    timestamp: Optional[float] = None
    min_below: int = Field(default=1, ge=1)
    min_over: int = Field(default=1, ge=1)
    test_ratio: Optional[float] = Field(default=None, ge=0.1, le=0.9)
    leave_n_out: Optional[int] = Field(default=None, ge=1)
    folds: int = Field(default=1, ge=1, le=20)

    @model_validator(mode="after")
    def validate_strategy_fields(self) -> "SplittingSingleConfig":
        """Validate conditional requirements based on the chosen splitting strategy.

        Returns:
            SplittingSingleConfig: The object itself.
        """
        match self.strategy:
            case SplittingStrategy.FIXED_TS:
                pass

            case SplittingStrategy.TEMP_HOLDOUT:
                if self.test_ratio is None and self.leave_n_out is None:
                    raise AttributeError(f"At least one among `test_ratio` and `leave_n_out` must be provided "
                                         f"with '{self.strategy.value}' strategy.")

            case SplittingStrategy.RAND_HOLDOUT:
                if self.test_ratio is None and self.leave_n_out is None:
                    raise AttributeError(f"At least one among `test_ratio` and `leave_n_out` must be provided "
                                         f"with '{self.strategy.value}' strategy.")

            case SplittingStrategy.RAND_SUB_SMP:
                if self.test_ratio is None and self.leave_n_out is None:
                    raise AttributeError(f"At least one among `test_ratio` and `leave_n_out` must be provided "
                                         f"with '{self.strategy.value}' strategy.")

            case SplittingStrategy.RAND_CV:
                min_val = 2
                if self.folds < min_val:
                    raise ValueError(f"Attribute `folds` must be at least {min_val} "
                                     f"with '{self.strategy.value}' strategy.")

        return self


class SplittingConfig(BaseConfig):
    """Splitting general configuration.

    Attributes:
        save_on_disk (bool): Whether to save split data to disk. Defaults to False.
        save_folder (str): Path to the folder where splits files will be saved (if `save_on_disk` is True).
        sequential (bool): Whether to save split as sequential or interactions data. Defaults to False.
        test_splitting (SplittingSingleConfig): Test splitting configuration.
        validation_splitting (SplittingSingleConfig, optional): Validation splitting configuration.
            Defaults to None.
        writer (Any): Writing configuration.
    """

    save_on_disk: bool = False
    save_folder: str = path_joiner("..", "data", "{0}", "splitting")
    sequential: bool = False
    test_splitting: SplittingSingleConfig
    validation_splitting: Optional[SplittingSingleConfig] = None
    writer: Any = Field(default={}, exclude=True)

    @model_validator(mode="after")
    def build_writer_config(self) -> "SplittingConfig":
        reader_cls = TabularWriterConfig if not self.sequential else SequenceWriterConfig
        self.writer = reader_cls(**self.writer)
        return self
