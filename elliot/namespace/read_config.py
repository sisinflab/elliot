from typing import List, Dict, Optional, Union
from pydantic import Field, field_validator, model_validator

from elliot.namespace.common import BaseConfig, normalize_ext, get_default_value


class Columns(BaseConfig):
    """Columns sub-configuration.

    Attributes:
        user_id_col (Union[str, int]): Name (or index) of the user ID column. Defaults to 'userId'.
        item_id_col (Union[str, int]): Name (or index) of the item ID column. Defaults to 'itemId'.
        rating_col (Union[str, int]): Name (or index) of the rating column. Defaults to 'rating'.
        timestamp_col (Union[str, int]): Name (or index) of the timestamp column. Defaults to 'timestamp'.
    """

    user_id_col: Union[str, int] = "userId"
    item_id_col: Union[str, int] = "itemId"
    rating_col: Union[str, int] = "rating"
    timestamp_col: Union[str, int] = "timestamp"


class Dtypes(BaseConfig):
    """Custom dtype sub-configuration.

    Attributes:
        user_id_type (str): The dtype to format the user_id column. Defaults to 'string'.
        item_id_type (str): The dtype to format the item_id column. Defaults to 'string'.
        rating_type (str): The dtype to format the rating column. Defaults to 'float'.
        timestamp_type (str): The dtype to format the timestamp column. Defaults to 'float'.
    """

    user_id_type: str = "string"
    item_id_type: str = "string"
    rating_type: str = "float"
    timestamp_type: str = "float"


class BaseReaderConfig(BaseConfig):
    """Base data reader configuration.

    Attributes:
        ext (List[str], optional): List of valid file extensions for reading. Defaults to None.
        patterns (str, optional): Filename patterns to match (e.g., "*.tsv"). Defaults to None.
    """

    ext: Optional[List[str]] = None
    patterns: Optional[str] = None

    @field_validator("ext", mode="after")
    @classmethod
    def normalize_ext(cls, v: List[str]) -> List[str]:
        """Validate and normalize a list of file extensions.

        Args:
            v (List[str]): List of file extensions to normalize.

        Returns:
            List[str]: Normalized list of file extensions.
        """
        normalized = set(normalize_ext(e) for e in v)
        return list(normalized)


class TabularReaderConfig(BaseReaderConfig):
    """Tabular data reader configuration.

    Attributes:
        header (bool): Whether the input file contains a header row. Defaults to False.
        columns (List[Union[str, int]], optional): List of column names or indices
            to select. Defaults to None.
        dtypes (Dict[Union[str, int], str], optional): Mapping of column names or indices
            to data types. Defaults to {}.
        sep (str): Column separator used in the input file. Defaults to "\\t".
        ext (List[str]): List of valid file extensions for reading. Defaults to [".tsv", ".csv"].
    """

    header: bool = False
    columns: Optional[List[Union[str, int]]] = None
    dtypes: Dict[Union[str, int], str] = {}
    sep: str = "\t"
    ext: List[str] = [".tsv", ".csv"]

    @model_validator(mode="after")
    def validate_dtypes(self) -> "TabularReaderConfig":
        if isinstance(self.dtypes, dict):
            self.dtypes = {k: v for k, v in self.dtypes if k in self.columns}
        return self


class InteractionsReaderConfig(TabularReaderConfig):
    """Interactions data reader configuration.

    Attributes:
        columns (Labels): Object containing column labels.
        dtypes (Dtypes): Object containing column dtypes.
    """

    columns: Columns = Field(default_factory=Columns)
    dtypes: Dtypes = Field(default_factory=Dtypes)

    def column_names(self) -> List[Union[str, int]]:
        """Return the list of names of the columns to read.

        Returns:
            List[Union[str, int]]: The list of column names.
        """
        return [
            self.columns.user_id_col,
            self.columns.item_id_col,
            self.columns.rating_col,
            self.columns.timestamp_col,
        ]

    def column_dtypes(self) -> Dict[Union[str, int], str]:
        """Return a dictionary mapping column names to their data types.

        Returns:
            Dict[Union[str, int], str]: A dictionary where keys are column names and values
                are corresponding data types.
        """
        column_names = self.column_names()
        column_dtypes = [
            self.dtypes.user_id_type,
            self.dtypes.item_id_type,
            self.dtypes.rating_type,
            self.dtypes.timestamp_type,
        ]
        return {
            name: dtype
            for name, dtype in zip(column_names, column_dtypes)
        }


class ModelReaderConfig(BaseReaderConfig):
    """Model reader configuration.

    Attributes:
        ext (List[str]): List of valid file extensions for reading. Defaults to [".pt", ".pth"].
    """

    ext: List[str] = [".pt", ".pth"]


class NumpyReaderConfig(BaseReaderConfig):
    """Numpy reader configuration.

    Attributes:
        ext (List[str]): List of valid file extensions for reading. Defaults to [".npy"].
    """

    ext: List[str] = [".npy"]


class GeneralReaderConfig(TabularReaderConfig, ModelReaderConfig, NumpyReaderConfig):
    """General reader configuration.

    Attributes:
        ext (List[str]): List of valid file extensions for reading.
            Defaults to [".tsv", ".csv", ".pt", ".pth", ".npy"].
    """

    ext: List[str] = (
        list(get_default_value(TabularReaderConfig, "ext") or []) +
        list(get_default_value(ModelReaderConfig, "ext") or []) +
        list(get_default_value(NumpyReaderConfig, "ext") or [])
    )
