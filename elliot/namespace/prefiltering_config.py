from typing import Optional, Union
from pydantic import Field, model_validator

from elliot.namespace.common import BaseConfig
from elliot.utils.enums import PreFilteringStrategy


class PreFilteringConfig(BaseConfig):
    """Pre-filtering configuration.

    Attributes:
        strategy (PreFilteringStrategy): Pre-filtering strategy to use.
        threshold (Union[float, int], optional): Threshold value for filtering. Defaults to None, min is 0.
        core (int): Core parameter for the strategy. Defaults to 5, min is 0.
        rounds (int): Number of rounds to perform. Defaults to 2, min is 0.
    """

    strategy: PreFilteringStrategy
    threshold: Optional[Union[float, int]] = Field(default=None, ge=0)
    core: int = Field(default=5, ge=0)
    rounds: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_strategy_fields(self) -> "PreFilteringConfig":
        """Ensure required fields are set for the selected pre-filtering strategy.

        Returns:
            PreFilteringConfig: The object itself.
        """
        if self.strategy == PreFilteringStrategy.COLD_USERS and self.threshold is None:
            raise AttributeError(f"Attribute `threshold` must be provided "
                                 f"with '{self.strategy.value}' strategy.")

        return self
