Evaluation of recommendation files
======================

Sometimes, the practitioner could need to evaluate an already computed recommendation file.
Either we forgot to involve some metrics or we want to compare our models with external baselines, Elliot provides
a facility to restore recommendation files and use them for the overall evaluation of the running experiment.

This is a sample config file with a proxy model restoring a recommendation file:

.. code:: yaml

    experiment:
        ...
        models:
            ProxyRecommender:
                path: path/to/recs/of/specific/model.tsv
            ItemKNN:
                ...


``ProxyRecommender`` is a fake recommender model which is able to restore old recommendation and prepare all
inner data structures to support Elliot evaluation pipeline. It plugs into the standard training/evaluation
pipeline as a regular ``Recommender`` (no custom training loop required).

ProxyRecommender options
------------------------

``ProxyRecommender`` accepts a few optional parameters to make file loading and evaluation more robust:

- ``sep``: delimiter used in the recommendation file (default: ``\\t``).
- ``header``: whether the file has a header row (``true``/``false``), or the header line index (default: ``false``).
- ``user_col`` / ``item_col`` / ``score_col``: column names or indices for user, item, and score.
  If ``score_col`` is omitted or missing, file order is used as a ranking signal.
- ``id_space``: ``public`` (default) or ``private`` if the file uses internal numeric ids.
- ``deduplicate``: keep the best score for duplicate user-item pairs (default: ``true``).
- ``filter_seen``: drop items already seen in the training set (and validation for test) (default: ``true``).
- ``strict``: raise errors on missing columns or unknown ids instead of skipping them (default: ``false``).

Example:

.. code:: yaml

    experiment:
        ...
        models:
            ProxyRecommender:
                path: path/to/recs/of/specific/model.tsv
                sep: "\t"
                header: false
                user_col: 0
                item_col: 1
                score_col: 2
                id_space: public
                deduplicate: true
                filter_seen: true
                strict: false
            ItemKNN:
                ...

Additionally, Elliot provides the practitioners with a facility to evaluate all the recommendation files stored in a folder.

This is a sample config file to restore recommendation files from a folder:

.. code:: yaml

    experiment:
        ...
        models:
            RecommendationFolder:
                folder: path/to/recs/folder
                pattern: "*.tsv"
                # or: extensions: [".tsv", ".csv"]
                sep: "\t"
                header: false
            ItemKNN:
                ...


``RecommendationFolder`` is a fake recommendation model that restores all the recommendation files found in the target ``folder`` and prepare all
inner data structures to support Elliot evaluation pipeline.

All ``ProxyRecommender`` options (e.g., ``sep``, ``header``, ``id_space``) can be specified under ``RecommendationFolder`` and will be applied to each file.
