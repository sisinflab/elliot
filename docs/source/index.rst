.. Elliot documentation master file, created by
   sphinx-quickstart on Mon Mar  1 15:05:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Elliot's documentation!
==================================

Elliot is a comprehensive recommendation framework that analyzes the
recommendation problem from the researcher's perspective. It conducts a
whole experiment, from dataset loading to results gathering. The core
idea is to feed the system with a simple and straightforward
configuration file that drives the framework through the experimental
setting choices. Elliot untangles the complexity of combining splitting
strategies, hyperparameter model optimization, model training, and the
generation of reports of the experimental results.

.. figure:: ../../img/elliot_img.png
   :alt: system schema

   system schema

The framework loads, filters, and splits the data considering a vast set
of strategies (splitting methods and filtering approaches, from temporal
training-test splitting to nested K-folds Cross-Validation). Elliot
optimizes hyperparameters for several recommendation algorithms, selects
the best models, compares them with the baselines providing intra-model
statistics, computes metrics spanning from accuracy to beyond-accuracy,
bias, and fairness, and conducts statistical analysis (Wilcoxon and
Paired t-test).

Elliot aims to keep the entire experiment reproducible and put the user
in control of the framework.


.. toctree::
   :maxdepth: 1
   :caption: GET STARTED

   guide/introduction
   guide/install
   guide/quick_start
   Release Notes <https://github.com/sisinflab/elliot/releases>

.. toctree::
   :maxdepth: 1
   :caption: RUNNING EXPERIMENTS

   guide/config
   guide/data_prep

.. toctree::
   :maxdepth: 1
   :caption: ALGORITHMS

   guide/alg_intro
   guide/hyper_optimization
   guide/new_alg
   guide/recommenders

.. toctree::
   :maxdepth: 1
   :caption: EVALUATION

   guide/metrics_intro
   guide/metrics_summary

.. toctree::
   :maxdepth: 1
   :caption: API REFERENCE

   elliot/elliot
   elliot/elliot.dataset
   elliot/elliot.evaluation
   elliot/elliot.namespace
   elliot/elliot.hyperoptimization
   elliot/elliot.prefiltering
   elliot/elliot.recommender
   elliot/elliot.result_handler
   elliot/elliot.splitter
   elliot/elliot.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
