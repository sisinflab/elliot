from typing import Literal, List, Optional, Union
from pydantic import field_validator

from elliot.namespace.common import BaseConfig, normalize_ext


class BaseWriterConfig(BaseConfig):
    """Base data writer configuration.

    Attributes:
        ext (str): File extension for the output file. Defaults to "".
        encoding (str, optional): File encoding. Defaults to None (using the platform default).
    """

    ext: str = ""
    encoding: Optional[str] = None

    @field_validator("ext", mode="after")
    @classmethod
    def normalize_ext(cls, v: str) -> str:
        """Validate and normalize a file extension.

        Args:
            v (str): File extension to normalize.

        Returns:
            str: Normalized file extension.
        """
        return normalize_ext(v)


class TabularWriterConfig(BaseWriterConfig):
    """Tabular data writer configuration.

    Attributes:
        header (bool): Whether to write a header row in the output file. Defaults to True.
            If a list of strings is given, it is assumed to be aliases for the column names.
        columns (List[Union[str, int]], optional): List of column names or indices
            to select. Defaults to None.
        sep (str): Column separator to use in the output file. Defaults to "\\t".
        ext (str): File extension for the output file. Defaults to ".tsv".
    """

    header: Union[bool, List[str]] = True
    columns: Optional[List[Union[str, int]]] = None
    sep: str = "\t"
    ext: str = ".tsv"


class SequenceWriterConfig(TabularWriterConfig):
    """Sequential interaction data writer configuration.

    Attributes:
        format (str): Layout of the output file, either 'wide' or 'inline'. Defaults to 'wide'.
        sequence_sep (str): Separator to use inside the serialized sequence string. Only used
            when `format` is 'inline'. Defaults to ' '.
    """

    format: Literal["wide", "inline"] = "wide"
    sequence_sep: str = " "


class ModelWriterConfig(BaseWriterConfig):
    """Model data writer configuration.

    Attributes:
        ext (str): File extension for the output file. Defaults to ".pth".
    """

    ext: str = ".pth"
