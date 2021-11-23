import os
import time
import math
import random
import numpy as np
from itertools import islice
from operator import itemgetter
from collections import OrderedDict, Counter
import pandas as pd
# from modules.auxiliar import TextColor, import_as_tsv
from multiprocessing import Process, Queue, cpu_count
import multiprocessing
multiprocessing.set_start_method("fork")


class UserFeatureMapper:
    def __init__(self, data, item_features_dict, predicate_mapping: pd.DataFrame, random_seed=42):
        np.random.seed(random_seed)

        self.user_pos_items = data

        self.predicate_mapping = predicate_mapping.set_index('predicate')['predicate_order'].to_dict()

        self.client_features = dict()
        self.clients_with_feature_extracted = set()
        self.item_features_dict = item_features_dict

    def build_item_features_dict(self):
        """
        For each item builds its features reading predicates and objects from the item_features dataset
        :return: dict {key = item_id : value = item features}
        """
        return {item: set(map(tuple,
                              self.item_features[self.item_features.itemId == self.item_ids[item]]
                              [['predicate', 'object']].values))
                for item in self.item_ids}

    def feature_counter(self, client, depth, neg_item_method='list'):
        """

        :param client:
        :param depth:
        :param neg_item_method:
        :return:
        """

        def pick_negative_uniform(positive):
            """
            Pick a negative item for each positive item.
            Every negative item in the training set is picked with the same probability
            :param positive: list of client positive items
            :return: set of picked negative items
            """

            items = set(self.item_features_dict)
            picked_negative = set()

            for _ in positive:
                negative = random.choice(items)
                while negative in neg_items or negative in positive:
                    negative = random.choice(items)
                picked_negative.add(negative)

            return neg_items

        def pick_negative_with_popularity(positive):
            """
            Pick a negative item for each positive item.
            Popular items have more chances to be picked.
            :param positive: list of client positive items
            :return: set of picked negative items
            """
            picked_negative = set()

            for _ in positive:
                negative = random.choice(list(self.item_features_dict))
                while negative in positive:
                    negative = random.choice(list(self.item_features_dict))
                picked_negative.add(negative)

            return picked_negative

        def count_features(positive, negative, feature_depth):
            """
            Given a list of positive and negative items retrieves all them features and then counts them.
            :param feature_depth: depth of the features
            :param positive: list positive items
            :param negative: list of negative items
            :return:
            """

            pos_to_add = []
            for i in positive:
                pos_to_add += list(
                    set(filter(lambda x: self.predicate_mapping[x[0]] == feature_depth, self.item_features_dict[i])))
            pos_counter = Counter(pos_to_add)

            neg_to_add = []
            for i in negative:
                neg_to_add += list(
                    set(filter(lambda x: self.predicate_mapping[x[0]] == feature_depth, self.item_features_dict[i])))
            neg_counter = Counter(neg_to_add)

            return pos_counter, neg_counter

        positive_items = self.user_pos_items[client]
        # NEG ITEMS SELECTION
        if neg_item_method == 'set':
            neg_items = pick_negative_uniform(positive_items)
        elif neg_item_method == 'list':
            neg_items = pick_negative_with_popularity(positive_items)
        else:
            neg_items = pick_negative_with_popularity(positive_items)

        counters = dict()
        for d in range(depth):
            pos_c, neg_c = count_features(positive_items, neg_items, d + 1)
            counters[d + 1] = (pos_c, neg_c, len(positive_items))

        return counters

    @staticmethod
    def features_entropy(pos_counter, neg_counter, counter):
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
            return - ratio * np.log2(ratio)

        def info_gain(pos_c, neg_c, n_items):

            den_1 = pos_c + neg_c
            h_pos = relative_gain(pos_c, den_1) + relative_gain(neg_c, den_1)
            den_2 = 2 * n_items - (pos_c + neg_c)

            num_1 = n_items - pos_c
            num_2 = n_items - neg_c
            h_neg = relative_gain(num_1, den_2) + relative_gain(num_2, den_2)

            return 1 - den_1 / (den_1 + den_2) * h_pos - den_2 / (den_1 + den_2) * h_neg

        attribute_entropies = dict()
        for positive_feature in pos_counter:
            ig = info_gain(pos_counter[positive_feature], neg_counter[positive_feature], counter)
            if ig > 0:
                attribute_entropies[positive_feature] = ig

        return OrderedDict(sorted(attribute_entropies.items(), key=itemgetter(1), reverse=True))

    def limited_second_order_selection(self, client, limit_first, limit_second, neg_item_method='list'):

        # positive and negative counters
        counters = self.feature_counter(client, 2, neg_item_method)

        # 1st-order-features
        pos_1, neg_1, counter_1 = counters[1]
        # 2nd-order-features
        pos_2, neg_2, counter_2 = counters[2]

        if limit_first == -1 and limit_second == -1:
            pos_f = pos_1 + pos_2
            neg_f = neg_1 + neg_2
        else:
            if limit_first == -1:
                if limit_second != 0:
                    entropies_2 = self.features_entropy(pos_2, neg_2, counter_2)
                    # top 2nd-order-features ordered by entropy
                    entropies_2_red = OrderedDict(islice(entropies_2.items(), limit_second))
                    # filtering pos and neg features respect to the 'top limit' selected
                    pos_2_red = Counter({k: pos_2[k] for k in entropies_2_red.keys()})
                    neg_2_red = Counter({k: neg_2[k] for k in entropies_2_red.keys()})
                else:
                    pos_2_red = Counter()
                    neg_2_red = Counter()

                # final features: 1st-order-f + top 2nd-order-f
                pos_f = pos_1 + pos_2_red
                neg_f = neg_1 + neg_2_red
            elif limit_second == -1:
                if limit_first != 0:
                    entropies_1 = self.features_entropy(pos_1, neg_1, counter_1)

                    # top 1st-order-features ordered by entropy
                    entropies_1_red = OrderedDict(islice(entropies_1.items(), limit_second))
                    # filtering pos and neg features respect to the 'top limit' selected
                    pos_1_red = Counter({k: pos_1[k] for k in entropies_1_red.keys()})
                    neg_1_red = Counter({k: neg_1[k] for k in entropies_1_red.keys()})
                else:
                    pos_1_red = Counter()
                    neg_1_red = Counter()

                # final features: top 1st-order-f + 2nd-order-f
                pos_f = pos_1_red + pos_2
                neg_f = neg_1_red + neg_2
            else:
                if limit_first != 0:
                    entropies_1 = self.features_entropy(pos_1, neg_1, counter_1)

                    # top 10 1st-order-features ordered by entropy
                    entropies_1_red = OrderedDict(islice(entropies_1.items(), limit_first))
                    # filtering pos and neg features respect to the 'top limit' selected
                    pos_1_red = Counter({k: pos_1[k] for k in entropies_1_red.keys()})
                    neg_1_red = Counter({k: neg_1[k] for k in entropies_1_red.keys()})
                else:
                    pos_1_red = Counter()
                    neg_1_red = Counter()

                if limit_second != 0:
                    entropies_2 = self.features_entropy(pos_2, neg_2, counter_2)

                    # top 10 2nd-order-features ordered by entropy
                    entropies_2_red = OrderedDict(islice(entropies_2.items(), limit_second))
                    # filtering pos and neg features respect to the 'top limit' selected
                    pos_2_red = Counter({k: pos_2[k] for k in entropies_2_red.keys()})
                    neg_2_red = Counter({k: neg_2[k] for k in entropies_2_red.keys()})
                else:
                    pos_2_red = Counter()
                    neg_2_red = Counter()

                # final features: top 1st-order-f + top 2nd-order-f
                pos_f = pos_1_red + pos_2_red
                neg_f = neg_1_red + neg_2_red

        return self.features_entropy(pos_f, neg_f, counter_2)

    def second_order_selection(self, client, neg_item_method='list'):

        # positive and negative counters
        counters = self.feature_counter(client, 2, neg_item_method)

        # 1st-order-features
        pos_1, neg_1, counter_1 = counters[1]
        # 2nd-order-features
        pos_2, neg_2, counter_2 = counters[2]

        # final features: 1st-order-f + top 10 2nd-order-f
        pos_f = pos_1 + pos_2
        neg_f = neg_1 + neg_2

        return self.features_entropy(pos_f, neg_f, counter_2)

    def compute_and_export_features(self, clients: list, parallel, first_order_limit, second_order_limit):

        def status_message(done: int, running: int):
            max_l = 25
            goal = len(clients)
            status = math.floor(done / goal * max_l)
            missing = max_l - status

            if done == goal:
                print('\rProgress:' + '|=' + '=' * status + '==' + '-' * missing + '|' +
                      f' {done} clients of {goal}')
                print(f'✓ DONE: extracted features of {goal} clients\n')
            else:
                print('\rProgress:' + '|=' + '=' * status + '=>' + '-' * missing + '|' +
                      f' [{done}/{goal} clients] - {running}/{parallel} processes running', end='')

        # num of parallel processes
        n_processes = parallel
        print(f'{n_processes} process')

        # create input chunks
        def split_in_chunks(l: list, n: int):
            """
            Split a list in a finite number of chunks.
            :param l: list to split
            :param n: number of chunks
            :return: a list of lists
            """

            list_l = len(l)
            residual = list_l % n
            chunk_size = int((list_l - residual) / n)
            chunks = [l[i: i + chunk_size] for i in range(0, list_l - residual, chunk_size)]
            chunks[0] += l[n * chunk_size:]
            return chunks

        chunks = split_in_chunks(clients, n_processes)

        processes = dict()
        output_q = Queue()

        # processes execution control
        active_processes = set()
        ended_processes = set()
        job_done_processes = set()

        # random seed for processes
        random_seed = np.random.randint(0, 100, n_processes)

        # set and run processes
        for pid in range(n_processes):
            signal_input_q = Queue()
            signal_output_q = Queue()
            # set process
            process = Process(
                target=self.compute_top_feature_worker,
                args=(chunks[pid],
                      signal_input_q,
                      signal_output_q,
                      output_q,
                      first_order_limit,
                      second_order_limit,
                      random_seed[pid]
                      )
            )
            processes[pid] = {'process': process,
                              'sig_i': signal_input_q, 'sig_o': signal_output_q,
                              'last': chunks[pid][-1]}

            # run process
            process.start()

        # init state: wait for processes to be ready
        while True:

            # round robin processes listening
            for pid in processes:
                queue: Queue = processes[pid]['sig_o']

                if queue.empty() is False:
                    message = queue.get()

                    # process is ready to start its job
                    if message == 'ready':
                        active_processes.add(pid)

            # if all processes are ready
            if len(active_processes) == n_processes:
                for pid in processes:
                    processes[pid]['sig_i'].put('start')
                break

        # running state: elaborate processes output
        while True:

            if output_q.empty() is False:
                # NEW CLIENT FEATURE
                p_output = output_q.get()
                self.client_features[p_output[0]] = p_output[1]
                self.clients_with_feature_extracted.add(p_output[0])

                for pid in processes:
                    if p_output[0] == processes[pid]['last']:
                        job_done_processes.add(pid)

            for pid in job_done_processes:
                if processes[pid]['sig_o'].empty() is False:
                    message = processes[pid]['sig_o'].get()
                    if message == 'done':
                        # print('done' + str(pid) + '\n')
                        active_processes.remove(pid)
                        # job_done_processes.remove(pid)
                        ended_processes.add(pid)

            # progress bar
            status_message(len(self.client_features), len(active_processes))

            # no more active processes, all job were done
            if len(active_processes) == 0:
                status_message(len(self.client_features), len(active_processes))
                # print('FINITO')
                break

        # end state: killing processes
        for pid in processes:
            processes[pid]['sig_i'].put('kill')

        time.sleep(1)

    def compute_top_feature_worker(self, client_list: list,
                                   signal_input: Queue, signal_output: Queue, output_q: Queue,
                                   first_order_limit, second_order_limit, random_seed):

        random.seed(random_seed)

        signal_output.put('ready')

        # init state
        while True:
            # look for new input
            if signal_input.empty() is False:
                message = signal_input.get()

                if message == 'start':
                    break

        # working state
        for client in client_list:
            features = self.limited_second_order_selection(client, first_order_limit, second_order_limit)
            output_q.put((client, features))

        signal_output.put('done')

        # job done state
        while True:
            time.sleep(1)
            # look for kill signal
            if signal_input.empty() is False:
                message = signal_input.get()
                if message == 'kill':
                    break

    def __getitem__(self, item):
        try:
            return self.client_features[item]
        except KeyError as e:
            print(f"You asked the features of client {item}, but its features have not been extracted.")

    def __iter__(self):
        return iter(self.client_features.items())
