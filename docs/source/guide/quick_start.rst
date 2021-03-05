Quick Start
======================

Hello Word Cofiguration
-----------------------

Elliot's entry point is the function ``run_experiment``, which accepts a
configuration file that drives the whole experiment. In the following, a
sample configuration file is shown to demonstrate how a sample and
explicit structure can generate a rigorous experiment.

.. code:: python

    from elliot.run import run_experiment

    run_experiment("configuration/file/path")

The following file is a simple configuration for an experimental setup.
It contains all the instructions to get the MovieLens-1M catalog from a
specific path and perform a train test split in a random sample way with
a ratio of 20%.

This experiment provides a hyperparameter optimization with a grid
search strategy for an Item-KNN model. Indeed, it is seen that the
possible values of neighbors are closed in squared brackets. It
indicates that two different models equipped with two different
neighbors' values will be trained and compared to select the best
configuration. Moreover, this configuration obliges Elliot to save the
recommendation lists with at most 10 items per user as suggest by top\_k
property.

In this basic experiment, only a simple metric is considered in the
final evaluation study. The candidate metric is nDCG for a cutoff equal
to top\_k, unless otherwise noted.

.. code:: yaml

    experiment:
      dataset: movielens_1m
      data_config:
        strategy: dataset
        dataset_path: ../data/movielens_1m/dataset.tsv
        splitting:
          test_splitting:
            strategy: random_subsampling
            test_ratio: 0.2
        models:
          ItemKNN:
            meta:
              hyper_opt_alg: grid
              save_recs: True
            neighbors: [50, 100]
            similarity: cosine
        evaluation:
          simple_metrics: [nDCG]
        top_k: 10


Basic Configuration
------------------------

In the first scenario, the experiments require comparing a group of RSs whose parameters are optimized via a grid-search.

The configuration specifies the data loading information, i.e., semantic features source files, in addition to the filtering and splitting strategies.

In particular, the latter supplies an entirely automated way of preprocessing the dataset, which is often a time-consuming
and non-easily-reproducible phase.

The simple_metrics field allows computing accuracy and beyond-accuracy metrics, with two top-k cut-off values (5 and 10)
by merely inserting the list of desired measures, e.g., [Precision, nDCG, ...].
The knowledge-aware recommendation model, AttributeItemKNN, is compared against two baselines: Random and ItemKNN,
along with a user-implemented model that is external.MostPop.

The configuration makes use of elliot's feature of conducting a grid search-based hyperparameter optimization strategy
by merely passing a list of possible hyperparameter values, e.g., neighbors: [50, 70, 100].

The reported models are selected according to nDCG@10.

**To see the full configuration file please visit the following** `link_basic <https://github.com/sisinflab/elliot/blob/master/config_files/basic_configuration.yml>`_

**To run the experiment use the following** `script_basic <https://github.com/sisinflab/elliot/blob/master/sample_basic.py>`_

Advanced Configuration
------------------------

The second scenario depicts a more complex experimental setting.
In the configuration, the user specifies an elaborate data splitting strategy, i.e., random_subsampling (for test splitting)
and random_cross_validation (for model selection), by setting few splitting configuration fields.

The configuration does not provide a cut-off value, and thus a top-k field value of 50 is assumed as the cut-off.

Moreover, the evaluation section includes the UserMADrating metric.

Elliot considers it as a complex metric since it requires additional arguments.

The user also wants to implement a more advanced hyperparameter tuning optimization. For instance, regarding NeuMF,
Bayesian optimization using Tree of Parzen Estimators is required (i.e., hyper_opt_alg: tpe) with a logarithmic uniform
sampling for the learning rate search space.

Moreover, Elliot allows considering complex neural architecture search spaces by inserting lists of tuples. For instance,
(32, 16, 8) indicates that the neural network consists of three hidden layers with 32, 16, and 8 units, respectively.


**To see the full configuration file please visit the following** `link_advanced <https://github.com/sisinflab/elliot/blob/master/config_files/advanced_configuration.yml>`_

**To run the experiment use the following** `script_advanced <https://github.com/sisinflab/elliot/blob/master/sample_advanced.py>`_
