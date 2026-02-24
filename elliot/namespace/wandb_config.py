from typing import Optional
from pydantic import model_validator

from elliot.namespace.common import BaseConfig


class WandBConfig(BaseConfig):
    """Optional Weights & Biases integration settings.

    Attributes:
        project (str, optional): W&B project name.
        run_name (str, optional): Custom run name prefix.
    """

    project: Optional[str] = None
    run_name: Optional[str] = None

    @model_validator(mode="after")
    def normalize_values(self) -> "WandBConfig":
        if isinstance(self.project, str):
            self.project = self.project.strip() or None
        if isinstance(self.run_name, str):
            self.run_name = self.run_name.strip() or None
        return self

    def enabled(self) -> bool:
        return bool(self.project)
