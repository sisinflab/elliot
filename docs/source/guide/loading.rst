Loading Data
======================

RSs experiments could require different data sources such as user-item feedback or additional side information,e.g., the
visual features of an item images. To fulfill these requirements, Elliot comes with different implementations of the
Loading module. Additionally, the user can design computationally expensive prefiltering and splitting procedures that
can be stored and loaded to save future computation. Data-driven extensions can handle additional data like visual
features, and semantic features extracted from knowledge graphs. Once a side-information-aware Loading module is chosen,
it filters out the items devoiding the required information to grant a fair comparison.

It is possible to enable a specific Loader by inserting the field ``side_information`` and passing:
    - ``dataloader`` the name of a specific loader
    - a list of possible file or folder which get side information accordingly to loader

An example can be:

.. code:: yaml

    experiment:
      data_config:
        strategy: fixed
        train_path: this/is/the/path.tsv
        test_path: this/is/the/path.tsv
        side_information:
            - dataloader: FeatureLoader1
            map: this/is/the/path.tsv

