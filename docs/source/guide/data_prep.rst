Data Preparation
======================

The first key component of the config file is the ``data_config`` section.

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

Data preparation Strategies
"""""""""""""""""""""""""""""""""
According to the kind of data we have, we can choose among three different loading strategies: ``dataset``, ``fixed``, ``hierarchy``.

``dataset`` assumes that the input data is NOT previously split in training, validation, and test set.
For this reason, ONLY if we adopt a dataset strategy we can later perform prefiltering and splitting operations.

``dataset`` takes just ONE default parameter: ``dataset_path``, which points to the stored dataset.

.. code:: yaml

    experiment:
      data_config:
        strategy: dataset
        dataset_path: this/is/the/path.tsv

``fixed`` strategy assumes that our data has been previously split into training/validation/test sets or training/test sets.
Since data is supposed as previously split, no further prefiltering and splitting operation is contemplated.

``fixed`` takes two mandatory parameters: ``train_path`` and ``test_path``, and one optional parameter, ``validation_path``.

.. code:: yaml

    experiment:
      data_config:
        strategy: fixed
        train_path: this/is/the/path.tsv
        validation_path: this/is/the/path.tsv
        test_path: this/is/the/path.tsv

The last strategy is ``hierarchy``.
``hierarchy`` is designed to load a dataset that has been previously split and filtered with Elliot.
Here, the data is assumed as split and no further prefiltering and splitting operations are needed.

``hierarchy`` takes one mandatory parameter, ``root_folder``, that points to the folder where we previously stored the split files.

.. code:: yaml

    experiment:
      data_config:
        strategy: hierarchy
        root_folder: this/is/the/path

.. toctree::
   :maxdepth: 1

   loading
   data_loaders
   prefiltering
   splitting
