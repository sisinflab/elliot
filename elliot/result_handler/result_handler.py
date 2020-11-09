

class ResultHandler:
    def __init__(self):
        self.multishot_recommenders = {}
        self.oneshot_recommenders = {}

    def add_multishot_recommender(self, obj):
        recommenders = {}
        name = obj._trials[0]["result"]["params"]["name"].split("_")[0]
        for recommender in obj._trials:
            rec = {}
            rec["name"] = rec["params"]["name"]
            rec["loss"] = recommender["result"]["loss"]
            rec["params"] = recommender["result"]["params"]
            rec["results"] = recommender["result"]["results"]
            recommenders[rec["params"]["name"]] = rec
        self.multishot_recommenders[name] = recommenders

    def add_oneshot_recommender(self, name, loss, params, results):
        rec = {}
        rec["name"] = name
        rec["loss"] = loss
        rec["params"] = params
        rec["results"] = results
        self.oneshot_recommenders[name] = rec
