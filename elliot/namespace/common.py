from typing import Any, Union, get_origin, get_args
from pydantic import BaseModel, ConfigDict, model_validator


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
            BaseConfig: The updated configuration object.
        """
        for key, value in (self.model_extra or {}).items():
            if not hasattr(self, key):
                setattr(self, key, value)
        return self


def check_type(annotation: Any, desired_type: type) -> bool:
    """Check whether an annotation contains a specific generic type.

    Support Union annotations by inspecting their generic arguments.

    Args:
        annotation (Any): Type annotation to inspect.
        desired_type (type): Generic type to check for (e.g., list, tuple).

    Returns:
        bool: True if the desired type is found, False otherwise.
    """
    origin = get_origin(annotation)
    if origin is Union:
        return any(get_origin(arg) is desired_type for arg in get_args(annotation))
    return get_origin(annotation) is desired_type


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
