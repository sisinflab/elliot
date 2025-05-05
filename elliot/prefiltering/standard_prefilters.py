import pandas as pd
from types import SimpleNamespace

"""
prefiltering:
    strategy: global_threshold|user_average|user_k_core|item_k_core|iterative_k_core|n_rounds_k_core|cold_users
    threshold: 3|average
    core: 5
    rounds: 2
"""


class PreFilter:

    _mask = None

    @staticmethod
    def filter(d: pd.DataFrame, ns: SimpleNamespace) -> pd.DataFrame:
        if not hasattr(ns, "prefiltering"):
            return d
        dataframe = d.copy()
        for strategy in ns.prefiltering:
            dataframe = PreFilter.single_filter(dataframe, strategy)
        return dataframe

    @staticmethod
    def single_filter(d: pd.DataFrame, ns: SimpleNamespace) -> pd.DataFrame:
        strategy = getattr(ns, "strategy", None)
        if not strategy:
            raise ValueError("Missing strategy")
        data = d.copy()

        match strategy:
            case "global_threshold":
                threshold = PreFilter._get_validated_attr(ns, "threshold", expected_type=(int, str))
                if isinstance(threshold, str):
                    if threshold == "average":
                        PreFilter._filter_ratings_by_global_average(data)
                    else:
                        raise ValueError(f"Invalid threshold value '{threshold}'")
                else:
                    PreFilter._filter_ratings_by_threshold(data, PreFilter._get_validated_attr(ns, "threshold"))

            case "user_average":
                PreFilter._filter_ratings_by_user_average(data)

            case "user_k_core":
                PreFilter._filter_users_by_profile_size(data, PreFilter._get_validated_attr(ns, "core"))

            case "item_k_core":
                PreFilter._filter_items_by_popularity(data, PreFilter._get_validated_attr(ns, "core"))

            case "iterative_k_core":
                return PreFilter._filter_iterative_k_core(data, PreFilter._get_validated_attr(ns, "core"))

            case "n_rounds_k_core":
                core = PreFilter._get_validated_attr(ns, "core")
                rounds = PreFilter._get_validated_attr(ns, "rounds")
                return PreFilter._filter_rounds_k_core(data, core, rounds)

            case "cold_users":
                PreFilter._filter_retain_cold_users(data, PreFilter._get_validated_attr(ns, "threshold"))

            case _:
                raise ValueError(f"Strategy '{strategy}' not recognized")

        return data[PreFilter._mask]

    @staticmethod
    def _get_validated_attr(ns, attr, expected_type=int, required=True):
        val = getattr(ns, attr, None)
        if required and val is None:
            raise ValueError(f"{attr} option is missing")

        if isinstance(expected_type, tuple):
            msg = f"{attr} must be an integer or a string"
        else:
            msg = f"{attr} must be an integer"
        if not isinstance(val, expected_type):
            raise ValueError(msg)

        return val

    @staticmethod
    def _filter_ratings_by_global_average(d: pd.DataFrame) -> None:
        data = d.copy()
        threshold = data["rating"].mean()
        PreFilter._mask = data['rating'] >= threshold
        print("\nPrefiltering with Global Average")
        print(f"The rating average is {round(threshold, 1)}")
        print(f"The transactions above threshold are {PreFilter._mask.sum()}")
        print(f"The transactions below threshold are {(~PreFilter._mask).sum()}")

    @staticmethod
    def _filter_ratings_by_threshold(d: pd.DataFrame, threshold) -> None:
        data = d.copy()
        PreFilter._mask = data['rating'] >= threshold
        print("\nPrefiltering with fixed threshold")
        print(f"The rating threshold is {round(threshold, 1)}")
        print(f"The transactions above threshold are {PreFilter._mask.sum()}")
        print(f"The transactions below threshold are {(~PreFilter._mask).sum()}\n")

    @staticmethod
    def _filter_ratings_by_user_average(d: pd.DataFrame) -> None:
        data = d.copy()
        PreFilter._mask = data['rating'] >= data.groupby('userId')['rating'].transform('mean')
        print("\nPrefiltering with user average")
        print(f"The transactions above threshold are {PreFilter._mask.sum()}")
        print(f"The transactions below threshold are {(~PreFilter._mask).sum()}\n")

    @staticmethod
    def _filter_users_by_profile_size(d: pd.DataFrame, threshold) -> None:
        data = d.copy()
        user_counts = data["userId"].value_counts()
        valid_users = user_counts[user_counts >= threshold].index
        PreFilter._mask = data["userId"].isin(valid_users)
        print(f"\nPrefiltering with user {threshold}-core")
        print(f"The transactions before filtering are {len(data)}")
        print(f"The users before filtering are {data['userId'].nunique()}")
        print(f"The transactions after filtering are {PreFilter._mask.sum()}")
        print(f"The users after filtering are {data[PreFilter._mask]['userId'].nunique()}")

    @staticmethod
    def _filter_items_by_popularity(d: pd.DataFrame, threshold) -> None:
        data = d.copy()
        item_counts = data["itemId"].value_counts()
        valid_items = item_counts[item_counts >= threshold].index
        PreFilter._mask = data["itemId"].isin(valid_items)
        print(f"\nPrefiltering with item {threshold}-core")
        print(f"The transactions before filtering are {len(data)}")
        print(f"The items before filtering are {data['itemId'].nunique()}")
        print(f"The transactions after filtering are {PreFilter._mask.sum()}")
        print(f"The items after filtering are {data[PreFilter._mask]['itemId'].nunique()}")

    @staticmethod
    def _filter_iterative_k_core(d: pd.DataFrame, threshold) -> pd.DataFrame:
        data = d.copy()
        original_length = -1
        print("\n**************************************")
        print(f"Iterative {threshold}-core")
        while original_length != len(data):
            original_length = len(data)
            PreFilter._filter_users_by_profile_size(data, threshold)
            data = data[PreFilter._mask]
            PreFilter._filter_items_by_popularity(data, threshold)
            data = data[PreFilter._mask]
        print("**************************************\n")

        return data

    @staticmethod
    def _filter_rounds_k_core(d: pd.DataFrame, threshold, n_rounds) -> pd.DataFrame:
        data = d.copy()
        print("\n**************************************")
        print(f"{n_rounds} rounds of user/item {threshold}-core")
        for i in range(n_rounds):
            print(f"Iteration:\t{i}")
            PreFilter._filter_users_by_profile_size(data, threshold)
            data = data[PreFilter._mask]
            PreFilter._filter_items_by_popularity(data, threshold)
            data = data[PreFilter._mask]
        print("**************************************\n")

        return data

    @staticmethod
    def _filter_retain_cold_users(d: pd.DataFrame, threshold) -> None:
        data = d.copy()
        user_counts = data["userId"].value_counts()
        cold_users = user_counts[user_counts <= threshold].index
        PreFilter._mask = data["userId"].isin(cold_users)
        print(f"\nPrefiltering retaining cold users with {threshold} or less ratings")
        print(f"The transactions before filtering are {len(data)}")
        print(f"The users before filtering are {data['userId'].nunique()}")
        print(f"The transactions after filtering are {PreFilter._mask.sum()}")
        print(f"The users after filtering are {data[PreFilter._mask]['userId'].nunique()}")
