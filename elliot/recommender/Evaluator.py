import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import math
from time import time
import datetime

_feed_dict = None
_dataset = None
_model = None
_sess = None
_K = None


def _init_eval_model(data):
    global _dataset
    _dataset = data

    pool = Pool(cpu_count() - 1)
    feed_dicts = pool.map(_evaluate_input, range(_dataset.num_users))
    pool.close()
    pool.join()

    return feed_dicts


def _evaluate_input(user):
    # generate items_list
    try:
        test_item = _dataset.test[user][1]
        item_input = set(range(_dataset.num_items)) - set(_dataset.train_list[user])
        if test_item in item_input:
            item_input.remove(test_item)
        item_input = list(item_input)
        item_input.append(test_item)
        user_input = np.full(len(item_input), user, dtype='int32')[:, None]
        item_input = np.array(item_input)[:, None]
        return user_input, item_input
    except:
        print('User '+str(user)+' is not present within the test set!')
        return 0, 0


def _eval_by_user(user, curr_pred):
    # get predictions of data in testing set
    user_input, item_input = _feed_dicts[user]
    if type(user_input) != np.ndarray:
        return ()
    predictions = curr_pred[list(item_input.reshape(-1))]
    neg_predict, pos_predict = predictions[:-1], predictions[-1]
    position = (neg_predict >= pos_predict).sum()

    # calculate from HR@1 to HR@10, and from NDCG@1 to NDCG@100, AUC
    hr, ndcg, auc = [], [], []
    for k in range(1, _K + 1):
        hr.append(position < k)
        ndcg.append(math.log(2) / math.log(position + 2) if position < k else 0)
        auc.append(
            1 - (position / len(neg_predict)))  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]

    return hr, ndcg, auc


class Evaluator:
    def __init__(self, model, data, k):
        """
        Class to manage all the evaluation methods and operation
        :param data: dataset object
        :param k: top-k evaluation
        """
        self.data = data
        self.k = k
        self.eval_feed_dicts = _init_eval_model(data)
        self.model = model

    def eval(self, epoch=0, results={}, epoch_text='', start_time=0):
        """
        Runtime Evaluation of Accuracy Performance (top-k)
        :return:
        """
        global _model
        global _K
        global _dataset
        global _feed_dicts
        _dataset = self.data
        _model = self.model
        _K = self.k
        _feed_dicts = self.eval_feed_dicts

        res = []

        eval_start_time = time()
        all_predictions = self.model.predict_all().numpy()

        for user in range(self.model.data.num_users):
            current_prediction = all_predictions[user, :]
            res.append(_eval_by_user(user, current_prediction))

        res = list(filter(None, res))
        hr, ndcg, auc = (np.array(res).mean(axis=0)).tolist()
        print("%s \tTrain Time: %s \tEval Time: %s \tPerformance@%d ==> HR: %.4f\tnDCG: %.4f\tAUC: %.4f" % (
            epoch_text,
            datetime.timedelta(seconds=(time() - start_time)),
            datetime.timedelta(seconds=(time() - eval_start_time)),
            _K,
            hr[_K - 1],
            ndcg[_K - 1],
            auc[_K - 1]))

        if len(epoch_text) != '':
            results[epoch] = {'hr': hr, 'ndcg': ndcg, 'auc': auc[0]}

    def store_recommendation(self, attack_name="", path=""):
        """
        Store recommendation list (top-k) in order to be used for the ranksys framework (anonymized)
        attack_name: The name for the attack stored file
        :return:
        """
        results = self.model.predict_all().numpy()
        with open(path, 'w') as out:
            for u in range(results.shape[0]):
                results[u][self.data.train_list[u]] = -np.inf
                top_k_id = results[u].argsort()[-self.k:][::-1]
                top_k_score = results[u][top_k_id]
                for i, value in enumerate(top_k_id):
                    out.write(str(u) + '\t' + str(value) + '\t' + str(top_k_score[i]) + '\n')