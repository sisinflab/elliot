import warnings
import pandas as pd
import typing as t
from types import SimpleNamespace


class PreFilter:
    """
    The PreFilter class applies various pre-filtering strategies to a user-item interaction dataset.

    This class allows filtering interactions based on different strategies, such as global thresholds,
    user/item k-core filtering, user-specific rating averages, and cold user retention.

    Attributes:
        data (pd.DataFrame): The input dataset, typically containing 'userId', 'itemId', and 'rating' columns.
        prefiltering_ns (List[SimpleNamespace]): A list of configurations specifying filtering strategies.
        _mask (pd.Series): Internal mask used during filtering steps.

    Supported Pre-filtering Strategies:
        - global_threshold: Removes interactions with ratings below a fixed threshold or global average.
        - user_average: Removes interactions with ratings below the user's average rating.
        - user_k_core: Retains users with at least `core` interactions.
        - item_k_core: Retains items with at least `core` interactions.
        - iterative_k_core: Iteratively applies user and item k-core filtering until convergence.
        - n_rounds_k_core: Applies k-core filtering for a fixed number of rounds.
        - cold_users: Retains only users with interactions less than or equal to a specified threshold.

    To configure the data pre-filtering, include the appropriate
    settings in the configuration file using the pattern shown below.
    Note: Pre-filtering is optional and can be applied regardless of the `data_config.strategy` value.

    .. code:: yaml

      prefiltering:
        - strategy: global_threshold|user_average|user_k_core|item_k_core|iterative_k_core|n_rounds_k_core|cold_users
          threshold: 3|average
          core: 5
          rounds: 2
        - strategy: global_threshold|user_average|user_k_core|item_k_core|iterative_k_core|n_rounds_k_core|cold_users
          threshold: 3|average
          core: 5
          rounds: 2
    """

    def __init__(self, data: pd.DataFrame, prefiltering_ns: t.List[SimpleNamespace]):
        """
        Initializes the PreFilter object to manage dataset pre-filtering strategies.

        Args:
            data (pd.DataFrame): The dataset to be split.
            prefiltering_ns (SimpleNamespace): A namespace object containing pre-filtering configuration.
        """
        self.data = data
        self.prefiltering_ns = prefiltering_ns
        self._mask = None

    def filter(self) -> pd.DataFrame:
        """
        Applies all configured pre-filtering strategies in sequence to the dataset.

        Returns:
            pd.DataFrame: The filtered dataset.
        """
        dataframe = self.data.copy()
        for strategy in self.prefiltering_ns:
            dataframe = self.single_filter(dataframe, strategy)
        return dataframe

    def single_filter(self, data: pd.DataFrame, ns: SimpleNamespace) -> pd.DataFrame:
        """
        Applies a single pre-filtering strategy to the dataset based on the provided configuration.

        Args:
            data (pd.DataFrame): The input dataset to be filtered.
            ns (SimpleNamespace): A namespace containing the strategy name and any required parameters.

        Returns:
            pd.DataFrame: The filtered dataset.

        Raises:
            ValueError: If the strategy is missing or unrecognized, or if required parameters are invalid.
        """
        strategy = getattr(ns, "strategy", None)
        if not strategy:
            raise ValueError("Missing strategy")

        filtered_data = None
        match strategy:
            case "global_threshold":
                threshold = self._get_validated_attr(ns, "threshold", expected_type=(int, str))
                if isinstance(threshold, str):
                    self.filter_ratings_by_global_average(data)
                else:
                    self.filter_ratings_by_threshold(data, self._get_validated_attr(ns, "threshold"))

            case "user_average":
                threshold = None
                self.filter_ratings_by_user_average(data)

            case "user_k_core":
                threshold = self._get_validated_attr(ns, "core")
                self.filter_user_k_core(data, threshold)

            case "item_k_core":
                threshold = self._get_validated_attr(ns, "core")
                self.filter_item_k_core(data, threshold)

            case "iterative_k_core":
                threshold = self._get_validated_attr(ns, "core")
                filtered_data = self.filter_iterative_k_core(data, threshold)

            case "n_rounds_k_core":
                threshold = self._get_validated_attr(ns, "core")
                rounds = self._get_validated_attr(ns, "rounds")
                filtered_data = self.filter_n_rounds_k_core(data, threshold, rounds)

            case "cold_users":
                threshold = self._get_validated_attr(ns, "threshold")
                self.filter_retain_cold_users(data, threshold)

            case _:
                raise ValueError(f"Strategy '{strategy}' not recognized")

        return self._apply_mask_and_check(data, strategy, threshold, filtered_data)

    def _get_validated_attr(self, ns, attr, expected_type=int, required=True):
        """
        Validates the presence and type of required attributes in the configuration namespace.

        Args:
           ns (SimpleNamespace): The configuration object.
           attr (str): The name of the attribute to validate.
           expected_type (type or tuple): Expected type(s) of the attribute.
           required (bool): Whether the attribute is mandatory.

        Returns:
           Any: The validated attribute value.

        Raises:
           ValueError: If the attribute is missing.
           TypeError: If the attribute has invalid type.
        """
        val = getattr(ns, attr, None)
        if required and val is None:
            raise ValueError(f"Missing required attribute: '{attr}'")
        if val is not None and not isinstance(val, expected_type):
            raise TypeError(f"Attribute '{attr}' must be of type {expected_type}, got {type(val).__name__}")

        # Additional value constraints
        if isinstance(val, int) and val < 0:
            raise ValueError(f"Attribute '{attr}' must be a non-negative integer")
        if isinstance(val, str) and val != 'average':
            raise ValueError(f"Attribute '{attr}' must be 'average' if a string is used")

        return val

    def _apply_mask_and_check(self, data, strategy, th, filtered_data=None):
        """
        Applies a boolean mask to the data and checks whether the resulting filtered dataset
        is sufficiently large to be considered valid. If not, returns the original data.

        Args:
            data (pd.DataFrame): The original dataset to be filtered.
            strategy (str): The name of the filtering strategy applied (used in warnings).
            th (Any): The threshold used for filtering (only used for reporting purposes).
            filtered_data (pd.DataFrame, optional): Pre-filtered data to use instead of applying
                the mask. If None, `self._mask` is applied to `data`.

        Returns:
            pd.DataFrame: The filtered dataset if the ratio of filtered rows is acceptable;
                otherwise, the original unfiltered dataset.

        Raises:
            Warning: Emits a warning if the filtered dataset is too small (less than 5% of
                original) or empty.
        """
        filtered_data = filtered_data if filtered_data is not None else data[self._mask]
        ratio = len(filtered_data)/len(data)
        if data.empty or ratio < 0.05:
            warnings.warn(f"Pre-filtering with strategy {strategy} ignored: "
                          f"dataset is empty or too reduced using threshold {th}.")
            return data
        else:
            return filtered_data

    def filter_ratings_by_global_average(self, data: pd.DataFrame) -> None:
        """
        Filters out ratings below the global average rating across the dataset.

        Args:
            data (pd.DataFrame): The input dataset.
        """
        threshold = data["rating"].mean()
        self._mask = data['rating'] >= threshold
        print("\nPrefiltering with Global Average")
        print(f"The rating average is {round(threshold, 1)}")
        print(f"The transactions above threshold are {self._mask.sum()}")
        print(f"The transactions below threshold are {(~self._mask).sum()}")

    def filter_ratings_by_threshold(self, data: pd.DataFrame, threshold) -> None:
        """
        Filters out ratings below a fixed threshold.

        Args:
            data (pd.DataFrame): The input dataset.
            threshold (float): The rating threshold.
        """
        self._mask = data['rating'] >= threshold
        print("\nPrefiltering with fixed threshold")
        print(f"The rating threshold is {round(threshold, 1)}")
        print(f"The transactions above threshold are {self._mask.sum()}")
        print(f"The transactions below threshold are {(~self._mask).sum()}\n")

    def filter_ratings_by_user_average(self, data: pd.DataFrame) -> None:
        """
        Filters out ratings that fall below each user's average rating.

        Args:
            data (pd.DataFrame): The input dataset.
        """
        self._mask = data['rating'] >= data.groupby('userId')['rating'].transform('mean')
        print("\nPrefiltering with user average")
        print(f"The transactions above threshold are {self._mask.sum()}")
        print(f"The transactions below threshold are {(~self._mask).sum()}\n")

    def filter_user_k_core(self, data: pd.DataFrame, threshold) -> None:
        """
        Retains only users with at least `threshold` interactions.

        Args:
            data (pd.DataFrame): The input dataset.
            threshold (int): The minimum number of interactions required per user.
        """
        user_counts = data["userId"].value_counts()
        valid_users = user_counts[user_counts >= threshold].index
        self._mask = data["userId"].isin(valid_users)
        print(f"\nPrefiltering with user {threshold}-core")
        print(f"The transactions before filtering are {len(data)}")
        print(f"The users before filtering are {data['userId'].nunique()}")
        print(f"The transactions after filtering are {self._mask.sum()}")
        print(f"The users after filtering are {data[self._mask]['userId'].nunique()}")

    def filter_item_k_core(self, data: pd.DataFrame, threshold) -> None:
        """
        Retains only items with at least `threshold` interactions.

        Args:
            data (pd.DataFrame): The input dataset.
            threshold (int): The minimum number of interactions required per item.
        """
        item_counts = data["itemId"].value_counts()
        valid_items = item_counts[item_counts >= threshold].index
        self._mask = data["itemId"].isin(valid_items)
        print(f"\nPrefiltering with item {threshold}-core")
        print(f"The transactions before filtering are {len(data)}")
        print(f"The items before filtering are {data['itemId'].nunique()}")
        print(f"The transactions after filtering are {self._mask.sum()}")
        print(f"The items after filtering are {data[self._mask]['itemId'].nunique()}")

    def filter_iterative_k_core(self, data: pd.DataFrame, threshold) -> pd.DataFrame:
        """
        Applies iterative k-core filtering by alternating between user and item filtering
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

    def filter_n_rounds_k_core(self, data: pd.DataFrame, threshold, n_rounds) -> pd.DataFrame:
        """
        Applies a fixed number of user/item k-core filtering rounds.

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

    def filter_retain_cold_users(self, data: pd.DataFrame, threshold) -> None:
        """
        Retains only 'cold' users, i.e., users with `threshold` or fewer interactions.

        Args:
            data (pd.DataFrame): The input dataset.
            threshold (int): The maximum number of interactions to be considered a cold user.
        """
        user_counts = data["userId"].value_counts()
        cold_users = user_counts[user_counts <= threshold].index
        self._mask = data["userId"].isin(cold_users)
        print(f"\nPrefiltering retaining cold users with {threshold} or less ratings")
        print(f"The transactions before filtering are {len(data)}")
        print(f"The users before filtering are {data['userId'].nunique()}")
        print(f"The transactions after filtering are {self._mask.sum()}")
        print(f"The users after filtering are {data[self._mask]['userId'].nunique()}")
