from typing import List, Dict, Literal, Optional, Union
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
        if isinstance(self.dtypes, dict) and self.columns is not None:
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


class SequenceColumns(BaseConfig):
    """Sequence columns sub-configuration.

    Attributes:
        user_id_col (Union[str, int]): Name (or index) of the user identifier column. Defaults to 'user'.
        sequence_col (Union[str, int]): Name (or index) of the column containing the serialized
            sequence. Only used when `format` is 'inline'. Defaults to 'sequence'.
        timestamp_col (Union[str, int], optional): Name (or index) of the timestamp column,
            if present. Only used when `format` is 'inline'. Defaults to None.
        meta_cols (List[Union[str, int]], optional): Additional metadata columns to keep.
            Only used when `format` is 'inline'. Defaults to None.
    """

    user_id_col: Union[str, int] = "userId"
    sequence_col: Union[str, int] = "sequence"
    timestamp_col: Optional[Union[str, int]] = "timestamp"
    meta_cols: Optional[List[Union[str, int]]] = None


class SequenceDtypes(BaseConfig):
    """Custom dtype sub-configuration for sequential data.

    Attributes:
        user_id_type (str): The dtype to format the user_id column. Defaults to 'string'.
        sequence_type (str): The dtype to format the raw sequence column. Only used
            when `format` is 'inline'. Defaults to 'string'.
        timestamp_type (str): The dtype to format the timestamp column. Defaults to 'float'.
    """

    user_id_type: str = "string"
    sequence_type: str = "string"
    timestamp_type: str = "float"


class SequenceReaderConfig(TabularReaderConfig):
    """Sequential interaction data reader configuration.

    Attributes:
        header (bool): Whether the input file contains a header row. Defaults to True.
        columns (SequenceColumns): Object containing the sequence column labels.
        dtypes (SequenceDtypes): Object containing sequence column dtypes.
        format (str): Layout of the input file, either 'wide' or 'inline'. Defaults to 'wide'.
        sequence_sep (str): Separator used inside the serialized sequence string. Only used
            when `format` is 'inline'. Defaults to ' '.
    """

    header: bool = True
    columns: SequenceColumns = Field(default_factory=SequenceColumns)
    dtypes: SequenceDtypes = Field(default_factory=SequenceDtypes)
    format: Literal["wide", "inline"] = "wide"
    sequence_sep: str = " "

    def column_names(self) -> List[Union[str, int]]:
        """Return the ordered list of names (or indices) of the columns to read.

        For `format="inline"`, the order is [user, sequence, timestamp, *meta]
        (timestamp and meta entries are only included when configured). For
        `format="wide"`, only the user column is returned, since the remaining
        tokens of each row are treated as the interaction sequence.

        Returns:
            List[Union[str, int]]: The list of column names or indices.
        """
        names = [self.columns.user_id_col]

        if self.format == "inline":
            names.append(self.columns.sequence_col)
            if self.columns.timestamp_col is not None:
                names.append(self.columns.timestamp_col)
            names.extend(self.columns.meta_cols or [])

        return names

    def column_dtypes(self) -> Dict[Union[str, int], str]:
        """Return a dictionary mapping column names (or indices) to their data types.

        Returns:
            Dict[Union[str, int], str]: A dictionary where keys are column names/indices
                and values are the corresponding data types.
        """
        column_names = self.column_names()
        column_dtypes = [self.dtypes.user_id_type]

        if self.format == "inline":
            column_dtypes.append(self.dtypes.sequence_type)
            if self.columns.timestamp_col is not None:
                column_dtypes.append(self.dtypes.timestamp_type)

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
