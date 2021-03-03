Quick Start
======================

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