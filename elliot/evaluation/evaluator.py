import numpy as np


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