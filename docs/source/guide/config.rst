Configuration file
======================


Input Data Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The first key component of the config file is the data_config section.

.. code:: yaml

    experiment:
      data_config:
        strategy: dataset|fixed|hierarchy
        dataloader: KnowledgeChainsLoader|DataSetLoader
        dataset_path: this/is/the/path.tsv
        root_folder: this/is/the/path
        train_path: this/is/the/path.tsv
        validation_path: this/is/the/path.tsv
        test_path: this/is/the/path.tsv
        side_information:
            feature_data: this/is/the/path.tsv
            map: this/is/the/path.tsv
            features: this/is/the/path.tsv
            properties: this/is/the/path.conf

In this section, we can define which input files and how they should be loaded.

In the following, we will consider as datasets, tab-separated-value files that contain one interaction per row, in the format:

``UserID`` ``ItemID`` ``Rating`` [ ``TimeStamp`` ]

where ``TimeStamp`` is optional.

Strategies
"""""""""""
According to the kind of data we have, we can choose among three different loading strategies: ``dataset``, ``fixed``, ``hierarchy``.

``dataset`` assumes that the input data is NOT previously split in training, validation, and test set.
For this reason, ONLY if we adopt a dataset strategy we can later perform prefiltering and splitting operations.

``fixed`` strategy assumes that our data has been previously split into training/validation/test sets or training/test sets.
Since data is supposed as previously split, no further prefiltering and splitting operation is contemplated.

The last strategy is ``hierarchy``.
``hierarchy`` is designed to load a dataset that has been previously split and filtered with Elliot.
Here, the data is assumed as split and no further prefiltering and splitting operations are needed.

``dataset`` takes just ONE default parameter: ``dataset_path``, which points to the stored dataset.

``fixed`` takes two mandatory parameters: ``train_path`` and ``test_path``, and one optional parameter, ``validation_path``.

``hierarchy`` takes one mandatory parameter, ``root_folder``, that points to the folder where we previously stored the split files.


Data Loaders
"""""""""""""""""
Within the data_config section, we can also enable data-specific Data Loaders.
Each Data Loader is designed to handle a specific kind of additional data.

It is possible to enable a Data Loader by inserting the field ``dataloader`` and passing the corresponding name.
For instance, the Visual Data Loader lets the user consider precomputed visual feature vectors or (inclusive) images.

To pass the required parameters to the Data Loader, we use a specific subsection, named ``side_information``.
There we can enable the required (by the specific Data Loader) fields and insert the corresponding values.

An example can be:


.. code:: yaml

    experiment:
      data_config:
        strategy: fixed
        dataloader: VisualLoader
        train_path: this/is/the/path.tsv
        test_path: this/is/the/path.tsv
        side_information:
            feature_data: this/is/the/path/to/features.npy

For further details regarding the Data Loaders, please refer to XXX


Data Prefiltering
"""""""""""""""""""""""

Elliot provides several prefiltering strategies.
To enable Prefiltering operations, we can insert the corresponding block into our config file:

.. code:: yaml

      prefiltering:
        strategy: global_threshold|user_average|user_k_core|item_k_core|iterative_k_core|n_rounds_k_core|cold_users
        threshold: 3|average
        core: 5
        rounds: 2

In detail, Elliot provides eight main prefiltering approaches: ``global_threshold``,
``user_average``, ``user_k_core``, ``item_k_core``, ``iterative_k_core``, ``n_rounds_k_core``, ``cold_users``.

``global_threshold`` assumes a single system-wise threshold to filter out irrelevant transactions.
``global_threshold`` takes one mandatory parameter, ``threshold``.
``threshold`` takes, as values, a **float** (ratings >= threshold will be kept), or the string *average*. With average, the system computes the global mean of the rating values and filters out all the ratings below.

``user_average`` has no parameters, and the system filters out the ratings below each user rating values mean.

``user_k_core`` filters out all the users with a number of transactions lower than the given k core.
It takes a parameter, ``core``, where the user passes an **int** corresponding to the desired value.

``item_k_core`` filters out all the items with a number of transactions lower than the given k core.
It takes a parameter, ``core``, where the user passes an **int** corresponding to the desired value.

``iterative_k_core`` runs iteratively user_k_core, and item_k_core until the dataset is no further modified.
It takes a parameter, ``core``, where the user passes an **int** corresponding to the desired value.

``n_rounds_k_core`` runs iteratively user_k_core, and item_k_core for a specified number of rounds.
It takes the first parameter, ``core``, where the user passes an **int** corresponding to the desired value.
It takes the second parameter, ``rounds``, where the user passes an **int** corresponding to the desired value.

``cold_users`` filters out all the users with a number of interactions higher than a given threshold.
It takes a parameter, ``threshold``, where the user passes an **int** corresponding to the desired value.

.. code:: yaml

      prefiltering:
        strategy: global_threshold|user_average|user_k_core|item_k_core|iterative_k_core|n_rounds_k_core|cold_users
        threshold: 3|average
        core: 5
        rounds: 2
      dataset: categorical_dbpedia_ml1m

Data Splitting
""""""""""""""""""
.. code:: yaml

      splitting:
        save_on_disk: True
        save_folder: ../data/{0}/splitting/
        test_splitting:
            strategy: fixed_timestamp|temporal_hold_out|random_subsampling|random_cross_validation
            timestamp: best|1609786061
            test_ratio: 0.2
            leave_n_out: 1
            folds: 5
        validation_splitting:
            strategy: fixed_timestamp|temporal_hold_out|random_subsampling|random_cross_validation
            timestamp: best|1609786061
            test_ratio: 0.2
            leave_n_out: 1
            folds: 5


Output Configuration
"""""""""""""""""""""""
.. code:: yaml

      path_output_rec_result: ../results/{0}/recs/
      path_output_rec_weight: ../results/{0}/weights/
      path_output_rec_performance: ../results/{0}/performance/
      path_logger_config: ./config/logger_config.yml
      path_log_folder: ../log/

Evaluation Configuration
"""""""""""""""""""""""""""""
.. code:: yaml

      top_k: 50
      evaluation:
        cutoff: 10
        simple_metrics: [ nDCG, Precision, Recall, ItemCoverage, HR, MRR, MAP, F1, Gini, SEntropy, EFD, EPC, AUC, GAUC, LAUC, MAE, MSE, RMSE]
        relevance_threshold: 1
        paired_ttest: True
        complex_metrics:
        - metric: DSC
          beta: 2
        - metric: SRecall
          feature_data: ../data/categorical_dbpedia_ml1m/map.tsv

GPU Acceleration
"""""""""""""""""
.. code:: yaml

      gpu: -1 # -1 is not use GPU

Recommendation Model Configuration
"""""""""""""""""""""""""""""""""""""""""
.. code:: yaml

      models:
        MostPop:
          meta:
            save_recs: True