from typing import Any, Union, get_origin, get_args, List, Optional
from pydantic import BaseModel, ConfigDict, model_validator, field_validator
from pydantic_core.core_schema import ValidationInfo


class BaseConfig(BaseModel):
    """Base configuration model.

    Extend Pydantic BaseModel to allow extra fields and dynamically
    expose them as class attributes.
    """

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def set_extra_to_attrs(self):
        """Attach extra configuration fields as object attributes.

        Returns:
            BaseConfig: The object itself with the new fields.
        """
        for key, value in (self.model_extra or {}).items():
            if not hasattr(self, key):
                setattr(self, key, value)
        return self

    @field_validator("*", mode="before")
    @classmethod
    def transform_to_list(cls, value: Any, info: ValidationInfo) -> List[Any]:
        """Ensure that a configuration value is represented as a list.

        Args:
            value (Any): Input value from the configuration.
            info (ValidationInfo): Pydantic field metadata.

        Returns:
            List[Any]: Value converted to a list.
        """
        field = cls.model_fields[info.field_name]
        hint = field.annotation

        if check_type(hint, list, inspect_union=False):
            if not isinstance(value, list) and value is not None:
                value = [value]

        return value


def check_type(annotation: Any, desired_type: type, inspect_union: bool = True) -> bool:
    """Check whether an annotation contains a specific generic type.

    Support Union annotations by optionally inspecting their generic arguments.

    Args:
        annotation (Any): Type annotation to inspect.
        desired_type (type): Generic type to check for (e.g., list, tuple).
        inspect_union (bool): Whether to inspect Union arguments. Defaults to True.

    Returns:
        bool: True if the desired type is found, False otherwise.
    """
    origin = get_origin(annotation)
    if (origin is Union and inspect_union) or origin is Optional:
        return any(get_origin(arg) is desired_type for arg in get_args(annotation))
    return origin is desired_type


def check_range(
    attr_name: str,
    attr_val: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float]
):
    """Check if a numeric attribute value is within a specified range.

    Args:
        attr_name (str): Name of the attribute being validated.
        attr_val (Union[int, float]): Value of the attribute to check.
        min_val (Union[int, float]): Minimum allowed value (inclusive).
        max_val (Union[int, float]): Maximum allowed value (inclusive).

    Raises:
        ValueError: If `attr_val` is not within the range [min_val, max_val].
    """
    if not min_val <= attr_val <= max_val:
        raise ValueError(f"Attribute `{attr_name}` must be between {min_val} and {max_val} "
                         f"for the provided dataset.")


def normalize_ext(ext: str) -> str:
    """Normalize a file extension.

    Args:
        ext (str): File extension to normalize.

    Returns:
        str: Normalized file extension.
    """
    ext = ext.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext
