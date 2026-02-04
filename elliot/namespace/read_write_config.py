from typing import List, Optional, Union, Dict

from pydantic import Field

from elliot.namespace.common import BaseConfig


class Labels(BaseConfig):
    """Definition of the label sub-configuration.

    This class reads and optionally overrides the default labels of the data.

    Attributes:
        user_id_label (Optional[str]): Name of the user ID label; default is 'user_id'.
        item_id_label (Optional[str]): Name of the item ID label; default is 'item_id'.
        rating_label (Optional[str]): Name of the rating label; default is 'rating'.
        timestamp_label (Optional[str]): Name of the timestamp label; default is 'timestamp'.
    """

    user_id_label: Union[str, int] = "userId"
    item_id_label: Union[str, int] = "itemId"
    rating_label: Union[str, int] = "rating"
    timestamp_label: Union[str, int] = "timestamp"


class Dtypes(BaseConfig):
    """Definition of the custom dtype sub-configuration.

    This class reads and optionally overrides default dtypes of the data.

    Attributes:
        user_id_type (Optional[str]): The dtype to format the user_id column; default is 'string'.
        item_id_type (Optional[str]): The dtype to format the item_id column; default is 'string'.
        rating_type (Optional[str]): The dtype to format the rating column; default is 'float'.
        timestamp_type (Optional[str]): The dtype to format the timestamp column; default is 'float'.
    """

    user_id_type: str = "string"
    item_id_type: str = "string"
    rating_type: str = "float"
    timestamp_type: str = "float"


class ReaderConfig(BaseConfig):
    """Data reader configuration.

    Attributes:
        header (bool): Whether the input file contains a header row; default is False.
        labels (Labels): Object containing column labels.
        ext (Union[List[str], str]): File extension(s) of the input data; default is ".tsv".
        sep (str): Field separator used in the input file; default is "\\t".
    """

    header: bool = False
    labels: Labels = Field(default_factory=Labels)
    dtypes: Dtypes = Field(default_factory=Dtypes)
    ext: Union[List[str], str] = ".tsv"
    sep: str = "\t"

    def column_names(self) -> List[str]:
        """Return the list of names of the columns to read.

        Returns:
            List[str]: The list of column names.
        """
        return [
            self.labels.user_id_label,
            self.labels.item_id_label,
            self.labels.rating_label,
            self.labels.timestamp_label,
        ]

    def column_dtypes(self) -> Dict[Union[str, int], str]:
        """

        Returns:
            Dict[str, np.dtype]: A list containing the dtype to use for data loading.
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


class WriterConfig(BaseConfig):
    """Data writer configuration.

    Attributes:
        header (bool): Whether to write a header row in the output file; default is False.
        ext (str): File extension of the output data; default is ".tsv".
        sep (str): Field separator used in the output file; default is "\\t".
    """

    header: bool = False
    ext: str = ".tsv"
    sep: str = "\t"
