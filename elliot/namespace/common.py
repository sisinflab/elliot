from typing import List, Any, Optional, Union, ClassVar, get_origin, get_args, Set, Callable
from logging import LoggerAdapter
from pydantic import BaseModel, ConfigDict, model_validator, field_validator
from pydantic_core.core_schema import ValidationInfo

from elliot.utils.logging import get_logger


class BaseConfig(BaseModel):
    """Base configuration model.

    Extend Pydantic BaseModel to allow extra fields and dynamically
    expose them as class attributes.

    Attributes:
        warn_on_extra_fields (ClassVar[bool]): Whether to log warnings for extra fields
            in the configuration. Defaults to False.
        logger (ClassVar[LoggerAdapter]): A logging instance. Defaults to `get_logger("__main__")`.
    """

    model_config = ConfigDict(extra="allow", strict=False)
    warn_on_extra_fields: ClassVar[bool] = False
    logger: ClassVar[LoggerAdapter] = get_logger("__main__")

    @model_validator(mode="after")
    def set_extra_to_attrs(self):
        """Attach extra configuration fields as object attributes.

        Returns:
            BaseConfig: The object itself with the new fields.
        """
        for key, value in (self.model_extra or {}).items():
            if self.warn_on_extra_fields:
                self.logger.warning(
                    f"Unknown field '{key}' in {self.__class__.__name__} configuration."
                )
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


def build_fields_from_annotations(
    cls: object,
    exclude: Set[str] = {},
    field_fn: Callable = None,
) -> dict:
    """Build Pydantic field definitions from class annotations.

    Args:
        cls (object): The class from which keeping the annotations.
        exclude (Set[str]): The fields to exclude from the annotations.
        field_fn (Callable): The function to customize the field type.

    Returns:
        dict: The extracted fields' dict.
    """
    fields = {}

    for name, hint in cls.__annotations__.items():
        # Skip attributes in the 'exclude' set
        if name in exclude:
            continue

        # Get default value
        default = getattr(cls, name, "__MISSING__")

        field_type = field_fn(hint) if field_fn is not None else hint
        fields[name] = (field_type, default) if default != "__MISSING__" else field_type

    return fields


def get_default_value(model_cls, field_name) -> Any:
    f = model_cls.model_fields[field_name]
    if f.default_factory is not None:
        return f.default_factory()
    if f.default is not None:
        return f.default
    return None
