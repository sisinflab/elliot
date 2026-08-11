from typing import Literal, Optional
from pydantic import field_validator, model_validator

from elliot.namespace.common import BaseConfig


class WandBConfig(BaseConfig):
    """Optional Weights & Biases integration settings.

    Attributes:
        mode (Literal["online", "offline", "disabled"]): W&B execution mode.
            Defaults to "disabled".
        project (str, optional): W&B project name.
            Required when mode is "online" or "offline".
        run_prefix (str, optional): Custom run name prefix.
    """
    mode: Literal["online", "offline", "disabled"] = "disabled"
    project: Optional[str] = None
    run_prefix: Optional[str] = None

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, value) -> Literal["online", "offline", "disabled"]:
        """Normalize mode input to lowercase before Literal validation.

        Args:
            value (Any): Raw field value from the configuration.

        Returns:
            Literal["online", "offline", "disabled"]: Parsed and validated field value."""
        value = value.strip().lower()
        return value

    @model_validator(mode="after")
    def normalize_and_validate(self) -> "WandBConfig":
        """Normalize string fields and validate mode-dependent requirements.

        Returns:
            WandBConfig: The object itself"""
        if isinstance(self.project, str):
            self.project = self.project.strip() or None
        if isinstance(self.run_prefix, str):
            self.run_prefix = self.run_prefix.strip() or None

        if self.mode in {"online", "offline"} and not self.project:
            raise ValueError(
                "`wandb.project` is required when `wandb.mode` is 'online' or 'offline'."
            )
        return self
