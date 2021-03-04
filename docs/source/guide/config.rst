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


Data Splitting
""""""""""""""""""
Elliot provides several splitting strategies.
To enable the splitting operations, we can insert the corresponding section:

.. code:: yaml

      splitting:
        save_on_disk: True
        save_folder: this/is/the/path/
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

Before deepening the splitting configurations, we can configure Elliot to save on disk the split files, once the splitting operation is completed.

To this extent, we can insert two fields into the section: ``save_on_disk``, and ``save_folder``.

``save_on_disk`` enables the writing process, and ``save_folder`` specifies the system location where to save the split files:

.. code:: yaml

      splitting:
        save_on_disk: True
        save_folder: this/is/the/path/

Now, we can insert one (or two) specific subsections to detail the train/test, and the train/validation splitting via the corresponding fields:
``test_splitting``, and ``validation_splitting``.
``test_splitting`` is clearly mandatory, while ``validation_splitting`` is optional.
Since the two subsections follow the same guidelines, here we detail ``test_splitting`` without loss of generality.

Elliot enables four splitting families: ``fixed_timestamp``, ``temporal_hold_out``, ``random_subsampling``, ``random_cross_validation``.

``fixed_timestamp`` assumes that there will be a specific timestamp to split prior interactions (train) and future interactions.
It takes the parameter ``timestamp``, that can assume one of two possible kind of values: a **long** corresponding to a specific timestamp, or the string *best* computed following Anelli et al. XX.

``temporal_hold_out`` relies on a temporal split of user transactions. The split can be realized following two different approaches: a *ratio-based* and a *leave-n-out-based* approach.
If we enable the ``test_ratio`` field with a **float** value, Elliot splits data retaining the last (100 * ``test_ratio``) % of the user transactions for the test set.
If we enable the ``leave_n_out`` field with an **int** value, Elliot retains the last ``leave_n_out`` transactions for the test set.

``random_subsampling`` generalizes random hold-out strategy.
It takes a ``test_ratio`` parameter with a **float** value to define the train/test ratio for user-based hold-out splitting.
Alternatively, it can take ``leave_n_out`` with an **int** value to define the number of transaction retained for the test set.
Moreover, the splitting operation can be repeated enabling the ``folds`` field and passing an **int**.
In that case, the overall splitting strategy corresponds to a user-based random subsampling strategy.

``random_cross_validation`` adopts a k-folds cross-validation splitting strategy.
It takes the parameter ``folds`` with an **int** value, that defines the overall number of folds to consider.

Dataset Name Configuration
""""""""""""""""""""""""""""
Elliot needs a MANDATORY field, ``dataset``, that identifies the name of the dataset used for the experiment. This information is used in the majority of the experimental steps, to identify the experiment and save the files correctly:

.. code:: yaml

    experiment:
      dataset: dataset_name

Output Configuration
"""""""""""""""""""""""
Elliot lets the user specify where to store specific output files: the recommendation lists, the model weights, the evaluation results, and the logs:

.. code:: yaml

      path_output_rec_result: this/is/the/path/
      path_output_rec_weight: this/is/the/path/
      path_output_rec_performance: this/is/the/path/
      path_log_folder: this/is/the/path/

``path_output_rec_result`` lets the user define the path to the folder to store the recommendation lists.

``path_output_rec_weight`` lets the user define the path to the folder to store the model weights.

``path_output_rec_performance`` lets the user define the path to the folder to store the evaluation results.

``path_log_folder`` lets the user define the path to the folder to store the logs.

If not provided, Elliot creates a *results* folder in the parent folder of the config file location.

Inside it, Elliot creates an experiment-specific folder with the name of the *dataset*, and there it creates the *recs/*, *weights/*, and *performance/* folders, respectively.

Moreover, Elliot creates a *log/* folder in the parent folder of the config file location.


Evaluation Configuration
"""""""""""""""""""""""""""""

Elliot provides several facilities to evaluate recommendation systems.
The majority of the evaluation techniques require the computation of user-specific recommendation lists (some techniques use recommendation systems to perform knowledge completion or other tasks).

To define the length of the user recommendation list, Elliot provides a specific mandatory field, ``top_k``, that takes an **int** representing the list length.

Beyond the former general definition, to specify the evaluation configuration, we can insert a specific section:

.. code:: yaml

      top_k: 50
      evaluation:
        cutoffs: [10, 5]
        simple_metrics: [ nDCG, Precision, Recall]
        relevance_threshold: 1
        paired_ttest: True
        wilcoxon_test: True
        complex_metrics:
        - metric: DSC
          beta: 2
        - metric: SRecall
          feature_data: this/is/the/path.tsv

In that section, we can detail the main characteristics of our experimental benchmark.

In particular, we can provide Elliot with the information regarding the metrics we want to compute.
According to the metrics definition, some of them might require additional parameters or files.
To make it easier for the user to pass metrics and optional arguments, Elliot partitions the metrics into simple_metrics and complex_metrics.

simple_metrics can be inserted as a field into the evaluation section, and it takes as a value the list of the metrics we want to compute.
In the simple metrics set, we find all the metrics that **DO NOT** require any other additional parameter or file:

simple metrics example

The majority of the evaluation metrics relies on the notions of *cut-off* and *relevance threshold*.

The cut-off is the maximum length of the recommendation list we want to consider when computing the metric (it could be different from the top k).
To pass cut-off values, we can enable the ``cutoffs`` field and pass a single value or a **list of values**. Elliot will compute the evaluation results for each considered cut-off.
If cutoffs field is not provided, ``top_k`` value is assumed as a cut-off.

The relevance threshold is the minimum value of the rating to consider a test transaction relevant for the evaluation process.
We can pass the relevance threshold value to the corresponding ``relevance_threshold`` field.
If not given, relevance_threshold is set to **0**.

The set of metrics that require additional arguments is referred to as ``complex_metrics``.
The inclusion of the metrics follows the syntax:

.. code:: yaml

      evaluation:
        complex_metrics:
        - metric: complex_metric_name_0
          parameter_0: 2
        - metric: complex_metric_name_1
          parameter_1: this/is/the/path.tsv

where *parameter_0* and *parameter_1* are metric-specific parameters of any kind.

For further details about the available metrics, please see the corresponding section XXX.

Finally, Elliot enables the computation of paired statistical hypothesis tests, namely, *Wilcoxon*, and *Student's paired t-tests*.

To enable them, we can insert the corresponding boolean fields into the evaluation section:

.. code:: yaml

      evaluation:
        paired_ttest: True
        wilcoxon_test: True

All the evaluation results are available in the *performance* folder at the end of the experiment.

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