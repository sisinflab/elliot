from typing import List, Optional, Union
from types import SimpleNamespace
import warnings
import pandas as pd

from elliot.utils.enums import PreFilteringStrategy
from elliot.utils.validation import PreFilteringValidator


class PreFilter:
    """The PreFilter class applies various pre-filtering strategies to a user-item interaction dataset.

    This class allows filtering interactions based on different strategies, such as global thresholds,
    user/item k-core filtering, user-specific rating averages, and cold user retention.

    Attributes:
        data (pd.DataFrame): The input dataset, typically containing 'userId', 'itemId', and 'rating' columns.
        pre_filtering_ns (List[SimpleNamespace]): A list of configurations specifying filtering strategies.

    Supported pre-filtering strategies:

    - `global_threshold`: Remove interactions with ratings below a fixed threshold or global average.
    - `user_average`: Remove interactions with ratings below the user's average rating.
    - `user_k_core`: Retain users with at least `core` interactions.
    - `item_k_core`: Retain items with at least `core` interactions.
    - `iterative_k_core`: Iteratively apply user and item k-core filtering until convergence.
    - `n_rounds_k_core`: Apply k-core filtering for a fixed number of rounds.
    - `cold_users`: Retain only users with interactions less than or equal to a specified threshold.

    To configure the data pre-filtering, include the appropriate
    settings in the configuration file using the pattern shown below.

    .. code:: yaml

      prefiltering:
        - strategy: global_threshold|user_average|user_k_core|item_k_core|iterative_k_core|n_rounds_k_core|cold_users
          threshold: 3|None
          core: 5
          rounds: 2
        - strategy: global_threshold|user_average|user_k_core|item_k_core|iterative_k_core|n_rounds_k_core|cold_users
          threshold: 3|None
          core: 5
          rounds: 2

    Notes:
        Pre-filtering is optional and can be applied regardless of the `data_config.strategy` value.
    """

    strategy: PreFilteringStrategy
    threshold: Optional[Union[float, int]] = None
    core: Optional[int] = 5
    rounds: Optional[int] = 2

    def __init__(self, data: pd.DataFrame, pre_filtering_ns: List[SimpleNamespace]):
        self.data = data
        self.pre_filtering_ns = pre_filtering_ns
        self._mask = None

    def filter(self) -> pd.DataFrame:
        """
        Apply all configured pre-filtering strategies in sequence to the dataset.

        Returns:
            pd.DataFrame: The filtered dataset.
        """
        dataframe = self.data
        for strategy in self.pre_filtering_ns:
            dataframe = self.single_filter(dataframe, strategy)
        return dataframe

    def single_filter(self, data: pd.DataFrame, ns: SimpleNamespace) -> pd.DataFrame:
        """Apply a single pre-filtering strategy to the dataset based on the provided configuration.

        Args:
            data (pd.DataFrame): The input dataset to be filtered.
            ns (SimpleNamespace): A namespace containing the strategy name and any required parameters.

        Returns:
            pd.DataFrame: The filtered dataset.

        Raises:
            ValueError: If the strategy is invalid.
        """

        validator = PreFilteringValidator(**vars(ns))
        validator.assign_to_original(self)

        filtered_data = None

        match self.strategy:
            case PreFilteringStrategy.GLOBAL_TH:
                if self.threshold is None:
                    self.filter_ratings_by_global_average(data)
                else:
                    self.filter_ratings_by_threshold(data, self.threshold)

            case PreFilteringStrategy.USER_AVG:
                self.filter_ratings_by_user_average(data)

            case PreFilteringStrategy.USER_K_CORE:
                self.filter_user_k_core(data, self.core)

            case PreFilteringStrategy.ITEM_K_CORE:
                self.filter_item_k_core(data, self.core)

            case PreFilteringStrategy.ITER_K_CORE:
                filtered_data = self.filter_iterative_k_core(data, self.core)

            case PreFilteringStrategy.N_ROUNDS_K_CORE:
                filtered_data = self.filter_n_rounds_k_core(data, self.core, self.rounds)

            case PreFilteringStrategy.COLD_USERS:
                self.filter_retain_cold_users(data, self.threshold)

        return self._apply_mask_and_check(data, self.threshold, filtered_data)

    def filter_ratings_by_global_average(self, data: pd.DataFrame):
        """Filter out ratings below the global average rating across the dataset.

        Args:
            data (pd.DataFrame): The input dataset.
        """
        threshold = data["rating"].mean()
        self._mask = data['rating'] >= threshold
        print("\nPre-filtering with Global Average")
        print(f"The rating average is {round(threshold, 1)}")
        print(f"The transactions above threshold are {self._mask.sum()}")
        print(f"The transactions below threshold are {(~self._mask).sum()}")

    def filter_ratings_by_threshold(self, data: pd.DataFrame, threshold: float):
        """Filter out ratings below a fixed threshold.

        Args:
            data (pd.DataFrame): The input dataset.
            threshold (float): The rating threshold.
        """
        self._mask = data['rating'] >= threshold
        print("\nPre-filtering with fixed threshold")
        print(f"The rating threshold is {round(threshold, 1)}")
        print(f"The transactions above threshold are {self._mask.sum()}")
        print(f"The transactions below threshold are {(~self._mask).sum()}\n")

    def filter_ratings_by_user_average(self, data: pd.DataFrame):
        """Filter out ratings that fall below each user's average rating.

        Args:
            data (pd.DataFrame): The input dataset.
        """
        self._mask = data['rating'] >= data.groupby('userId')['rating'].transform('mean')
        print("\nPre-filtering with user average")
        print(f"The transactions above threshold are {self._mask.sum()}")
        print(f"The transactions below threshold are {(~self._mask).sum()}\n")

    def filter_user_k_core(self, data: pd.DataFrame, threshold: int):
        """Retain only users with at least `threshold` interactions.

        Args:
            data (pd.DataFrame): The input dataset.
            threshold (int): The minimum number of interactions required per user.
        """
        user_counts = data["userId"].value_counts()
        valid_users = user_counts[user_counts >= threshold].index
        self._mask = data["userId"].isin(valid_users)
        print(f"\nPre-filtering with user {threshold}-core")
        print(f"The transactions before filtering are {len(data)}")
        print(f"The users before filtering are {data['userId'].nunique()}")
        print(f"The transactions after filtering are {self._mask.sum()}")
        print(f"The users after filtering are {data[self._mask]['userId'].nunique()}")

    def filter_item_k_core(self, data: pd.DataFrame, threshold: int):
        """Retain only items with at least `threshold` interactions.

        Args:
            data (pd.DataFrame): The input dataset.
            threshold (int): The minimum number of interactions required per item.
        """
        item_counts = data["itemId"].value_counts()
        valid_items = item_counts[item_counts >= threshold].index
        self._mask = data["itemId"].isin(valid_items)
        print(f"\nPre-filtering with item {threshold}-core")
        print(f"The transactions before filtering are {len(data)}")
        print(f"The items before filtering are {data['itemId'].nunique()}")
        print(f"The transactions after filtering are {self._mask.sum()}")
        print(f"The items after filtering are {data[self._mask]['itemId'].nunique()}")

    def filter_iterative_k_core(self, data: pd.DataFrame, threshold: int) -> pd.DataFrame:
        """Apply iterative k-core filtering by alternating between user and item filtering
        until convergence.

        Args:
            data (pd.DataFrame): The input dataset.
            threshold (int): The minimum number of interactions per user and per item.

        Returns:
            pd.DataFrame: The filtered dataset.
        """
        original_length = -1
        print("\n**************************************")
        print(f"Iterative {threshold}-core")
        while original_length != len(data):
            original_length = len(data)
            self.filter_user_k_core(data, threshold)
            data = data[self._mask]
            self.filter_item_k_core(data, threshold)
            data = data[self._mask]
        print("**************************************\n")

        return data

    def filter_n_rounds_k_core(self, data: pd.DataFrame, threshold: int, n_rounds: int) -> pd.DataFrame:
        """Apply a fixed number of user/item k-core filtering rounds.

        Args:
            data (pd.DataFrame): The input dataset.
            threshold (int): The minimum number of interactions required.
            n_rounds (int): The number of iterations to perform.

        Returns:
            pd.DataFrame: The filtered dataset.
        """
        print("\n**************************************")
        print(f"{n_rounds} rounds of user/item {threshold}-core")
        for i in range(n_rounds):
            print(f"Iteration:\t{i}")
            self.filter_user_k_core(data, threshold)
            data = data[self._mask]
            self.filter_item_k_core(data, threshold)
            data = data[self._mask]
        print("**************************************\n")

        return data

    def filter_retain_cold_users(self, data: pd.DataFrame, threshold: int):
        """Retain only 'cold' users, i.e., users with `threshold` or fewer interactions.

        Args:
            data (pd.DataFrame): The input dataset.
            threshold (int): The maximum number of interactions to be considered a cold user.
        """
        user_counts = data["userId"].value_counts()
        cold_users = user_counts[user_counts <= threshold].index
        self._mask = data["userId"].isin(cold_users)
        print(f"\nPre-filtering retaining cold users with {threshold} or less ratings")
        print(f"The transactions before filtering are {len(data)}")
        print(f"The users before filtering are {data['userId'].nunique()}")
        print(f"The transactions after filtering are {self._mask.sum()}")
        print(f"The users after filtering are {data[self._mask]['userId'].nunique()}")

    def _apply_mask_and_check(
        self,
        data: pd.DataFrame,
        th: Union[int, float],
        filtered_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Apply a boolean mask to the data and checks whether the resulting filtered dataset
        is sufficiently large to be considered valid. If not, returns the original data.

        Args:
            data (pd.DataFrame): The original dataset to be filtered.
            th (Union[int, float]): The threshold used for filtering (only used for reporting purposes).
            filtered_data (Optional[pd.DataFrame]): Pre-filtered data to use instead of applying
                the mask. If None, `self._mask` is applied to `data`.

        Returns:
            pd.DataFrame: The filtered dataset if the ratio of filtered rows is acceptable;
                otherwise, the original unfiltered dataset.

        Raises:
            Warning: Emit a warning if the filtered dataset is too small (less than 60% of
                original) or empty.
        """
        filtered_data = filtered_data if filtered_data is not None else data[self._mask]
        ratio = len(filtered_data) / len(data)
        if data.empty or ratio < 0.6:
            warnings.warn(f"Pre-filtering with strategy {self.strategy.value} ignored: "
                          f"dataset is empty or too reduced using threshold {th}.")
            return data
        else:
            return filtered_data
