
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

    @staticmethod
    def filter(d: pd.DataFrame, ns: SimpleNamespace) -> pd.DataFrame:
        if not hasattr(ns, "prefiltering"):
            return d
        ns = ns.prefiltering
        dataframe = d.copy()

        for strategy in ns:
            dataframe = PreFilter.single_filter(dataframe, strategy)

        return dataframe

    @staticmethod
    def single_filter(d: pd.DataFrame, ns: SimpleNamespace) -> pd.DataFrame:

        strategy = getattr(ns, "strategy", None)
        data = d.copy()
        if strategy == "global_threshold":
            threshold = getattr(ns, "threshold", None)
            if threshold is not None:
                if str(threshold).isdigit():
                    data = PreFilter.filter_ratings_by_threshold(data, threshold)
                elif threshold == "average":
                    data = PreFilter.filter_ratings_by_global_average(data)
                else:
                    raise Exception("Threshold value not recognized")
            else:
                raise Exception("Threshold option is missing")

        elif strategy == "user_average":
            data = PreFilter.filter_ratings_by_user_average(data)

        elif strategy == "user_k_core":
            core = getattr(ns, "core", None)
            if core is not None:
                if str(core).isdigit():
                    data = PreFilter.filter_users_by_profile_size(data, core)
                else:
                    raise Exception("Core option is not a digit")
            else:
                raise Exception("Core option is missing")

        elif strategy == "item_k_core":
            core = getattr(ns, "core", None)
            if core is not None:
                if str(core).isdigit():
                    data = PreFilter.filter_items_by_popularity(data, core)
                else:
                    raise Exception("Core option is not a digit")
            else:
                raise Exception("Core option is missing")

        elif strategy == "iterative_k_core":
            core = getattr(ns, "core", None)
            if core is not None:
                if str(core).isdigit():
                    data = PreFilter.filter_iterative_k_core(data, core)
                else:
                    raise Exception("Core option is not a digit")
            else:
                raise Exception("Core option is missing")

        elif strategy == "n_rounds_k_core":
            core = getattr(ns, "core", None)
            n_rounds = getattr(ns, "rounds", None)
            if (core is not None) and (n_rounds is not None):
                if str(core).isdigit() and str(n_rounds).isdigit():
                    data = PreFilter.filter_rounds_k_core(data, core, n_rounds)
                else:
                    raise Exception("Core or rounds options are not digits")
            else:
                raise Exception("Core or rounds options are missing")

        elif strategy == "cold_users":
            threshold = getattr(ns, "threshold", None)
            if threshold is not None:
                if str(threshold).isdigit():
                    data = PreFilter.filter_retain_cold_users(data, threshold)
                else:
                    raise Exception("Threshold option is not a digit")
            else:
                raise Exception("Threshold option is missing")

        else:
            raise Exception("Misssing strategy")

        return data

    @staticmethod
    def filter_ratings_by_global_average(d: pd.DataFrame) -> pd.DataFrame:
        data = d.copy()
        threshold = data["rating"].mean()
        print("\nPrefiltering with Global Average")
        print(f"The rating average is {round(threshold, 1)}")
        print(f"The transactions above threshold are {data[data['rating'] >= threshold]['rating'].count()}")
        print(f"The transactions below threshold are {data[data['rating'] < threshold]['rating'].count()}")
        return data[data["rating"] >= threshold]

    @staticmethod
    def filter_ratings_by_threshold(d: pd.DataFrame, threshold) -> pd.DataFrame:
        data = d.copy()
        print("\nPrefiltering with fixed threshold")
        print(f"The rating threshold is {round(threshold, 1)}")
        print(f"The transactions above threshold are {data[data['rating'] >= threshold]['rating'].count()}")
        print(f"The transactions below threshold are {data[data['rating'] < threshold]['rating'].count()}\n")
        return data[data["rating"] >= threshold]

    @staticmethod
    def filter_ratings_by_user_average(d: pd.DataFrame) -> pd.DataFrame:
        data = d.copy()
        user_groups = data.groupby(['userId'])
        for name, group in user_groups:
            threshold = group["rating"].mean()
            data.loc[group.index, 'accept_flag'] = data.loc[group.index, 'rating'] >= threshold

        print("\nPrefiltering with user average")
        print(f"The transactions above threshold are {data[data['accept_flag']]['rating'].count()}")
        print(f"The transactions below threshold are {data[data['accept_flag'] == False]['rating'].count()}\n")
        return data[data["accept_flag"] == True].drop(columns=["accept_flag"]).reset_index(drop=True)

    @staticmethod
    def filter_users_by_profile_size(d: pd.DataFrame, threshold) -> pd.DataFrame:
        data = d.copy()
        print(f"\nPrefiltering with user {threshold}-core")
        print(f"The transactions before filtering are {len(data)}")
        print(f"The users before filtering are {data['userId'].nunique()}")
        user_groups = data.groupby(['userId'])
        data = user_groups.filter(lambda x: len(x) >= threshold)
        print(f"The transactions after filtering are {len(data)}")
        print(f"The users after filtering are {data['userId'].nunique()}")
        return data

    @staticmethod
    def filter_items_by_popularity(d: pd.DataFrame, threshold) -> pd.DataFrame:
        data = d.copy()
        print(f"\nPrefiltering with item {threshold}-core")
        print(f"The transactions before filtering are {len(data)}")
        print(f"The items before filtering are {data['itemId'].nunique()}")
        item_groups = data.groupby(['itemId'])
        data = item_groups.filter(lambda x: len(x) >= threshold)
        print(f"The transactions after filtering are {len(data)}")
        print(f"The items after filtering are {data['itemId'].nunique()}")
        return data

    @staticmethod
    def filter_iterative_k_core(d: pd.DataFrame, threshold) -> pd.DataFrame:
        data = d.copy()
        check_var = True
        original_length = len(data)
        print("\n**************************************")
        print(f"Iterative {threshold}-core")
        while check_var:
            data = PreFilter.filter_users_by_profile_size(data, threshold)
            data = PreFilter.filter_items_by_popularity(data, threshold)
            new_length = len(data)
            if original_length == new_length:
                check_var = False
            else:
                original_length = new_length
        print("**************************************\n")

        return data

    @staticmethod
    def filter_rounds_k_core(d: pd.DataFrame, threshold, n_rounds) -> pd.DataFrame:
        data = d.copy()
        print("\n**************************************")
        print(f"{n_rounds} rounds of user/item {threshold}-core")
        for i in range(n_rounds):
            print(f"Iteration:\t{i}")
            data = PreFilter.filter_users_by_profile_size(data, threshold)
            data = PreFilter.filter_items_by_popularity(data, threshold)
        print("**************************************\n")

        return data

    @staticmethod
    def filter_retain_cold_users(d: pd.DataFrame, threshold) -> pd.DataFrame:
        data = d.copy()
        print(f"\nPrefiltering retaining cold users with {threshold} or less ratings")
        print(f"The transactions before filtering are {len(data)}")
        print(f"The users before filtering are {data['userId'].nunique()}")
        user_groups = data.groupby(['userId'])
        data = user_groups.filter(lambda x: len(x) <= threshold)
        print(f"The transactions after filtering are {len(data)}")
        print(f"The users after filtering are {data['userId'].nunique()}")
        return data


# import unittest
#
#
#
# class NameSpaceModelTest(unittest.TestCase):
#     def setUp(self):
#         self.column_names = ['userId', 'itemId', 'rating']
#         self.data = pd.read_csv("../../data/example/trainingset.tsv", sep="\t", header=None, names=self.column_names)
#
#     def test_global_average(self):
#         data = PreFilter.filter_ratings_by_global_average(self.data)
#
#     def test_threshold(self):
#         data = PreFilter.filter_ratings_by_threshold(self.data, 3)
#
#     def test_user_average(self):
#         data = PreFilter.filter_ratings_by_user_average(self.data)
#
#     def test_user_k_core(self):
#         data = PreFilter.filter_users_by_profile_size(self.data, 10)
#
#     def test_item_k_core(self):
#         data = PreFilter.filter_items_by_popularity(self.data, 10)
#
#     def test_iterative_k_core(self):
#         data = PreFilter.filter_iterative_k_core(self.data, 10)
#
#     def test_rounds_k_core(self):
#         data = PreFilter.filter_rounds_k_core(self.data, 10, 2)
#
#     def test_cold_users(self):
#         data = PreFilter.filter_retain_cold_users(self.data, 10)
