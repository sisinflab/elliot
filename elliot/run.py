import importlib

from hyperopt import Trials, fmin

from namespace.namespace_model_builder import NameSpaceBuilder
import hyperoptimization as ho
import numpy as np

_rstate = np.random.RandomState(42)

if __name__ == '__main__':

    builder = NameSpaceBuilder('./config/config.yml')
    base = builder.base

    for key, model_base in builder.models():
        model_class = getattr(importlib.import_module("recommender"), key)
        if isinstance(model_base, tuple):
            model_placeholder = ho.ModelCoordinator(base.base_namespace, model_base[0], model_class)
            trials = Trials()
            best = fmin(model_placeholder.objective,
                        space=model_base[1],
                        algo=model_base[3],
                        trials=trials,
                        rstate=_rstate,
                        max_evals=model_base[2])

            print(best)
            min_val = np.argmin([i["result"]["loss"] for i in trials._trials])
            best_model_loss = trials._trials[min_val]["result"]["loss"]
            best_model_params = trials._trials[min_val]["result"]["params"]
            best_model_results = trials._trials[min_val]["result"]["results"]
        else:
            model = model_class(config=base.base_namespace, params=model_base)
            model.train()
            best_model_loss = model.get_loss()
            best_model_params = model.get_params()
            best_model_results = model.get_results()

        print(f"Loss: {best_model_loss}")
        print(f"Best Model params: {best_model_params}")
        print(f"Best Model results: {best_model_results}")
        print(f"\nHyperparameter tuning ended for {model_class.__name__}")
        print("********************************\n")

