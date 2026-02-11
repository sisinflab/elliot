from typing import List, Optional, Union
from pydantic import field_validator

from elliot.namespace.common import BaseConfig, normalize_ext


class BaseWriterConfig(BaseConfig):
    """Base data writer configuration.

    Attributes:
        ext (str): File extension for the output file. Defaults to "".
    """

    ext: str = ""

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


class ModelWriterConfig(BaseWriterConfig):
    """Model data writer configuration.

    Attributes:
        ext (str): File extension for the output file. Defaults to ".pth".
    """

    ext: str = ".pth"
