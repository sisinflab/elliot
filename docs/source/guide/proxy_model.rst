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
inner data structures to support Elliot evaluation pipeline.

Additionally, Elliot provides the practitioners with a facility to evaluate all the recommendation files stored in a folder.

This is a sample config file to restore recommendation files from a folder:

.. code:: yaml

    experiment:
        ...
        models:
            RecommendationFolder:
                folder: path/to/recs/folder
            ItemKNN:
                ...


``RecommendationFolder`` is a fake recommendation model that restores all the recommendation files found in the target ``folder`` and prepare all
inner data structures to support Elliot evaluation pipeline.

