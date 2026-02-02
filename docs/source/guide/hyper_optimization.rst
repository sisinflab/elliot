Hyperparameter Optimization
================================

Elliot provides hyperparameter tuning optimization integrating the functionalities of the **HyperOpt** library and supporting exhaustive grid search for discrete choices.

Before continuing, let us recall how to include a recommendation system into an experiment:

.. code:: yaml

    experiment:
      models:
        PMF:
          meta:
            hyper_max_evals: 20
            hyper_opt_alg: tpe
            validation_rate: 1
            verbose: True
            save_weights: True
            save_recs: True
            validation_metric: nDCG@10
          lr: 0.0025
          epochs: 2
          factors: 50
          batch_size: 512
          reg: 0.0025
          reg_b: 0
          gaussian_variance: 0.1

As we can observe, the *meta* section contains two fields that are related to hyperparameter optimization: ``hyper_max_evals``, and ``hyper_opt_alg``.


``hyper_opt_alg`` is a **string** field that defines the hyperparameter tuning strategy.

``hyper_opt_alg`` can assume one of the following values: *grid*, *tpe, *atpe*, *rand*, *mix*, and *anneal*.

*grid* corresponds to exhaustive grid search (discrete choices only)

*tpe* stands for Tree of Parzen Estimators, a type of Bayesian Optimization, see the `paper <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`_

*atpe* stands for Adaptive Tree of Parzen Estimators

*rand* stands for random sampling in the search space

*mix* stands for mixture of search algorithms

*anneal* stands for simulated annealing



``hyper_max_evals`` is an **int** field that defines the number of samples to consider for hyperparameter evaluation.
It is **required** when using Hyperopt distributions (e.g., ``uniform``, ``loguniform``). For *grid*, if omitted, Elliot
evaluates the full grid; if provided, it must be greater than or equal to the grid size.


Once we choose the search strategy, we need to define the search space.
To this end, Elliot provides two alternatives: a **value list**, and a **function-parameters pair**.

In the former case, we just need to provide a list of values to the parameter we want to optimize:

.. code:: yaml

    experiment:
      models:
        PMF:
          meta:
            hyper_max_evals: 20
            hyper_opt_alg: tpe
          lr:                   0.0025
          epochs:               2
          factors:              50
          batch_size:           512
          reg:                  [0.0025, 0.005, 0.01]
          reg_b:                0
          gaussian_variance:    0.1

In the latter case, we can choose among the search space functions provided by HyperOpt: *choice*, *randint*, *uniform*, *quniform*, *loguniform*, *qloguniform*, *normal*, *qnormal*, *lognormal*, *qlognormal*.
Each function and its parameters are documented at the `page <http://hyperopt.github.io/hyperopt/getting-started/search_spaces/>`_ in the section Parameters Expression.

Note that the label argument is internal and DO NOT have to provide it.

To teach Elliot to sample from any of these search spaces is straightforward: we pass to the parameter a list in which the first element is the function name, and the others are the parameter values.

An example of the syntax to define a search with *loguniform* for the learning rate parameter (lr) is:

.. code:: yaml

    experiment:
      models:
        PMF:
          meta:
            hyper_max_evals: 20
            hyper_opt_alg: tpe
          lr:                   [loguniform, -10, -1]
          epochs:               2
          factors:              50
          batch_size:           512
          reg:                  [0.0025, 0.005, 0.01]
          reg_b:                0
          gaussian_variance:    0.1

Finally, to perform an exhaustive grid search, set ``hyper_opt_alg: grid`` and provide discrete value lists for the
parameters to optimize:

.. code:: yaml

    experiment:
      models:
        PMF:
          meta:
            hyper_opt_alg: grid
            validation_rate: 1
            verbose: True
            save_weights: True
            save_recs: True
            validation_metric: nDCG@10
          lr:                   [0.0025, 0.005, 0.01]
          epochs:               50
          factors:              [10, 50, 100]
          batch_size:           512
          reg:                  [0.0025, 0.005, 0.01]
          reg_b:                0
          gaussian_variance:    0.1

In this case, Elliot performs an exhaustive grid search over the discrete combinations.

Note: Hyperopt trials are evaluated on validation metrics. After the search, the best configuration is retrained and
evaluated on the test split.
