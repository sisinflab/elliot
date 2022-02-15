import random
import math
import multiprocessing as mp
from tqdm import tqdm
from itertools import islice
from operator import itemgetter
from collections import OrderedDict, Counter

#mp.set_start_method("fork")
DEFAULT_FEATURE_WEIGHT = 'info_gain'


def worker(user, user_items, neg_items, if1, if2, fol, sol, npr, seed):
    random.seed(seed)
    counters = dict()
    counters[1] = user_features_counter(user_items, neg_items, if1, npr)
    counters[2] = user_features_counter(user_items, neg_items, if2, npr)
    return user, limited_second_order_selection(counters, fol, sol, npr)


class UserFeatureMapper:
    def __init__(self, data, item_features: dict, item_features2=None,
                 first_order_limit=100, second_order_limit=100, negative_positive_ratio=1):
        # for all the users compute the information for each feature
        self._data = data
        self._item_features = item_features
        self._item_features2 = item_features2
        self._first_order_limit = first_order_limit
        self._second_order_limit = second_order_limit
        self._users = self._data.private_users.keys()
        self._items = set(self._data.private_items.keys())
        self._depth = 2
        self._npr = negative_positive_ratio

        self.users_features = self.user_features_selected_MP()

    def user_features_selected_MP(self):
        def args():
            return ((u,
                     set(self._data.i_train_dict[u].keys()),
                     set.difference(self._items, set(self._data.i_train_dict[u].keys())),
                     self._item_features, self._item_features2,
                     self._first_order_limit,
                     self._second_order_limit,
                     self._npr,
                     random.randint(0, 100000)) for u in self._users)

        # arguments = args()
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     results = pool.starmap(worker, tqdm(arguments, total=len(self._users),
        #                                         desc="Computing user features and entropy..."))

        arguments = args()
        pool = mp.Pool(processes=mp.cpu_count())
        results = pool.starmap(worker,
                               tqdm(arguments, total=len(self._users), desc="Computing user features and entropy..."))
        pool.close()
        pool.join()

        return {u: f for u, f in results}


def user_features_counter(user_items, neg_items, item_features_, npr):
    def count_features(item_features_):
        """
        Given a list of positive and negative items retrieves all them features and then counts them.
        :param positive: list positive items
        :param negative: list of negative items
        :param item_features_:
        :return:
        """

        pos_features = []
        for p in positives:
            pos_features.extend(item_features_.get(p, set()))

        neg_features = []
        for n in negatives:
            neg_features.extend(item_features_.get(n, set()))

        pos_counter = Counter(pos_features)
        neg_counter = Counter(neg_features)

        return pos_counter, neg_counter

    # for each postive item pick a negative
    negatives = random.choices(list(neg_items), k=npr * len(user_items))
    positives = user_items

    # count positive feature and negative features
    pos_c, neg_c = count_features(item_features_)
    #return pos_c, neg_c, len(positives), len(negatives)
    return pos_c, neg_c, len(positives), len(negatives)


def feature_entropy(pos_counter, neg_counter, n_pos_items, n_neg_items, npr):
    """
    :param pos_counter: number of times in which feature is true and target is true
    :param neg_counter: number of times in which feature is true and target is false
    :param counter: number of items from which feaures have been extracted
    :return: dictionary feature: entropy with descending order by entropy
    """

    def relative_gain(partial, total):
        if total == 0:
            return 0
        ratio = partial / total
        if ratio == 0:
            return 0
        return - ratio * math.log2(ratio)

    def info_gain(pos_c, neg_c, n_pos_items, n_neg_items):

        den_1 = pos_c + neg_c

        h_pos = relative_gain(pos_c, den_1) + relative_gain(neg_c, den_1)
        den_2 = n_pos_items + n_neg_items - (pos_c + neg_c)

        num_1 = n_pos_items - pos_c
        num_2 = n_neg_items - neg_c
        h_neg = relative_gain(num_1, den_2) + relative_gain(num_2, den_2)

        return 1 - den_1 / (den_1 + den_2) * h_pos - den_2 / (den_1 + den_2) * h_neg

    attribute_entropies = dict()
    for positive_feature in pos_counter:
        ig = info_gain(pos_counter[positive_feature]*npr, neg_counter[positive_feature], n_pos_items*npr, n_neg_items)
        if ig > 0:
            attribute_entropies[positive_feature] = ig
    for negative_feature in neg_counter:
        ig = info_gain(pos_counter[negative_feature]*npr, neg_counter[negative_feature], n_pos_items*npr, n_neg_items)
        if ig > 0:
            attribute_entropies[negative_feature] = ig

    return OrderedDict(sorted(attribute_entropies.items(), key=itemgetter(1, 0), reverse=True))

def feature_gini(pos_counter, neg_counter, n_pos_items, n_neg_items):
    # TODO: change var names according to the formula
    def relative_gini(partial, total):
        if total == 0:
            return 0
        return (partial / total) ** 2

    def gini_index(pos_c, neg_c, n_pos_items, n_neg_items):

        den_1 = pos_c + neg_c
        gini_pos = 1 - (relative_gini(pos_c, den_1) + relative_gini(neg_c, den_1))

        den_2 = n_pos_items + n_neg_items - (pos_c + neg_c)
        num_1 = n_pos_items - pos_c
        num_2 = n_neg_items - neg_c
        gini_neg = 1 - (relative_gini(num_1, den_2) + relative_gini(num_2, den_2))

        return 1 - (den_1 / (den_1 + den_2) * gini_pos + den_2 / (den_1 + den_2) * gini_neg)

    attribute_entropies = dict()
    # compute Gini for each feature

    # let's multiply the value of pos_counter to obtain a weighted version of gini impurity
    ratio = n_neg_items / n_pos_items

    for positive_feature in pos_counter:
        gini = gini_index(pos_counter[positive_feature] * ratio, neg_counter[positive_feature], ratio * n_pos_items,
                          n_neg_items)
        attribute_entropies[positive_feature] = gini
    for negative_feature in neg_counter:
        gini = gini_index(pos_counter[negative_feature] * ratio, neg_counter[negative_feature], ratio * n_pos_items,
                          n_neg_items)
        attribute_entropies[negative_feature] = gini

    return OrderedDict(sorted(attribute_entropies.items(), key=itemgetter(1, 0), reverse=True))

# def bak_limited_second_order_selection(counters, limit_first, limit_second):
#     # 1st-order-features
#     pos_1, neg_1, counter_1 = counters[1]
#     # 2nd-order-features
#     pos_2, neg_2, counter_2 = counters[2]
#
#     if limit_first == -1 and limit_second == -1:
#         pos_f = pos_1 + pos_2
#         neg_f = neg_1 + neg_2
#     else:
#         if limit_first == -1:
#             if limit_second != 0:
#                 entropies_2 = feature_weight(pos_2, neg_2, counter_2)
#                 # top 2nd-order-features ordered by entropy
#                 entropies_2_red = OrderedDict(islice(entropies_2.items(), limit_second))
#                 # filtering pos and neg features respect to the 'top limit' selected
#                 pos_2_red = Counter({k: pos_2[k] for k in entropies_2_red.keys()})
#                 neg_2_red = Counter({k: neg_2[k] for k in entropies_2_red.keys()})
#             else:
#                 pos_2_red = Counter()
#                 neg_2_red = Counter()
#
#             # final features: 1st-order-f + top 2nd-order-f
#             pos_f = pos_1 + pos_2_red
#             neg_f = neg_1 + neg_2_red
#         elif limit_second == -1:
#             if limit_first != 0:
#                 entropies_1 = feature_weight(pos_1, neg_1, counter_1)
#
#                 # top 1st-order-features ordered by entropy
#                 entropies_1_red = OrderedDict(islice(entropies_1.items(), limit_second))
#                 # filtering pos and neg features respect to the 'top limit' selected
#                 pos_1_red = Counter({k: pos_1[k] for k in entropies_1_red.keys()})
#                 neg_1_red = Counter({k: neg_1[k] for k in entropies_1_red.keys()})
#             else:
#                 pos_1_red = Counter()
#                 neg_1_red = Counter()
#
#             # final features: top 1st-order-f + 2nd-order-f
#             pos_f = pos_1_red + pos_2
#             neg_f = neg_1_red + neg_2
#         else:
#             if limit_first != 0:
#                 entropies_1 = feature_weight(pos_1, neg_1, counter_1)
#
#                 # top 10 1st-order-features ordered by entropy
#                 entropies_1_red = OrderedDict(islice(entropies_1.items(), limit_first))
#                 # filtering pos and neg features respect to the 'top limit' selected
#                 pos_1_red = Counter({k: pos_1[k] for k in entropies_1_red.keys()})
#                 neg_1_red = Counter({k: neg_1[k] for k in entropies_1_red.keys()})
#             else:
#                 pos_1_red = Counter()
#                 neg_1_red = Counter()
#
#             if limit_second != 0:
#                 entropies_2 = feature_weight(pos_2, neg_2, counter_2)
#
#                 # top 10 2nd-order-features ordered by entropy
#                 entropies_2_red = OrderedDict(islice(entropies_2.items(), limit_second))
#                 # filtering pos and neg features respect to the 'top limit' selected
#                 pos_2_red = Counter({k: pos_2[k] for k in entropies_2_red.keys()})
#                 neg_2_red = Counter({k: neg_2[k] for k in entropies_2_red.keys()})
#             else:
#                 pos_2_red = Counter()
#                 neg_2_red = Counter()
#
#             # final features: top 1st-order-f + top 2nd-order-f
#             pos_f = pos_1_red + pos_2_red
#             neg_f = neg_1_red + neg_2_red
#
#     return feature_weight(pos_f, neg_f, counter_2)


def limited_second_order_selection(counters, limit_first, limit_second, npr):
    # 1st-order-features
    pos_1, neg_1, num_pos_items_1, num_neg_items_1 = counters[1]
    # 2nd-order-features
    pos_2, neg_2, num_pos_items_2, num_neg_items_2 = counters[2]

    entropies_1 = feature_weight(pos_1, neg_1, num_pos_items_1, num_neg_items_1, npr)
    entropies_2 = feature_weight(pos_2, neg_2, num_pos_items_2, num_neg_items_2, npr)
    entropies = {**entropies_1, **entropies_2}

    attributes_dict = OrderedDict(islice(sorted(entropies.items(), key=itemgetter(1, 0), reverse=True), limit_first))

    # mn = min(entropies.values())
    # mx = max(entropies.values())
    # diff = mx - mn
    #
    # if diff:
    #     for k, v in entropies.items():
    #         entropies[k] = (v - mn) / diff

    return attributes_dict

def feature_weight(pos_counter, neg_counter, num_pos_items, num_neg_items, npr, type=DEFAULT_FEATURE_WEIGHT):
    if type == 'info_gain':
        return feature_entropy(pos_counter, neg_counter, num_pos_items, num_neg_items, npr)
    elif type == 'gini':
        return feature_gini(pos_counter, neg_counter, num_pos_items, num_neg_items)
