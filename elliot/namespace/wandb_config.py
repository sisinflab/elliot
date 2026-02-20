from typing import Optional
from pydantic import model_validator

from elliot.namespace.common import BaseConfig


class WandBConfig(BaseConfig):
    """Optional Weights & Biases integration settings.

    Attributes:
        project (str, optional): W&B project name.
        api_key (str, optional): W&B API key.
    """

    project: Optional[str] = None
    api_key: Optional[str] = None

    @model_validator(mode="after")
    def normalize_values(self) -> "WandBConfig":
        if isinstance(self.project, str):
            self.project = self.project.strip() or None
        if isinstance(self.api_key, str):
            self.api_key = self.api_key.strip() or None
        return self

    def enabled(self) -> bool:
        return bool(self.project and self.api_key)
