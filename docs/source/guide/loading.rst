Loading Data
======================

RSs experiments could require different data sources such as user-item feedback or additional side information,e.g., the
visual features of an item images. To fulfill these requirements, Elliot comes with different implementations of the
Loading module. Additionally, the user can design computationally expensive prefiltering and splitting procedures that
can be stored and loaded to save future computation. Data-driven extensions can handle additional data like visual
features, and semantic features extracted from knowledge graphs. Once a side-information-aware Loading module is chosen,
it filters out the items devoiding the required information to grant a fair comparison.

Within the ``data_config`` section, we can also enable data-specific Data Loaders.
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

