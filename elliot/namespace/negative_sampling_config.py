from typing import Optional
from pydantic import Field, model_validator

from elliot.namespace.common import BaseConfig
from elliot.namespace.read_write_config import ReaderConfig, WriterConfig
from elliot.utils.enums import NegativeSamplingStrategy
from elliot.utils.folder import path_joiner


class NegativeSamplingConfig(BaseConfig):
    """Negative sampling configuration.

    Attributes:
        strategy (NegativeSamplingStrategy): Negative sampling strategy to use.
        num_negatives (int): Number of negative samples; default is 99, min is 1.
        leave_one_out (bool): Whether to add only one positive item to the sampled negatives per user; default is False.
        save_on_disk (bool): Whether to save sampling results to disk; default is False.
        save_folder (Optional[str]): Folder path to save negative samples.
        read_folder (Optional[str]): Folder containing negative samples files; required for `fixed` strategy.
        reader (ReaderConfig): Reading configuration.
        writer (WriterConfig): Writing configuration.
    """

    strategy: NegativeSamplingStrategy
    num_negatives: int = Field(default=99, ge=1)
    leave_one_out: bool = False
    save_on_disk: bool = False
    save_folder: Optional[str] = path_joiner("..", "data", "{0}")
    read_folder: Optional[str] = None
    reader: ReaderConfig = Field(default_factory=ReaderConfig, exclude=True)
    writer: WriterConfig = Field(default_factory=WriterConfig, exclude=True)

    @model_validator(mode="after")
    def validate_strategy_fields(self) -> "NegativeSamplingConfig":
        """Ensure required fields are set for the selected negative sampling strategy.

        Returns:
            NegativeSamplingConfig: The object itself.
        """
        if self.strategy == NegativeSamplingStrategy.FIXED and self.read_folder is None:
            raise AttributeError(f"Attribute `read_folder` must be provided "
                                 f"with '{self.strategy.value}' strategy.")

        return self
