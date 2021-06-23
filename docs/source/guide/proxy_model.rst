Proxy Model
======================

Elliot offers a lot of metrics to evaluate system's performance. Could be happen that during the experiment we forgot to
involve some metrics or we have to compare the recommendations' model with respect to other baselines. Elliot provides
a facilities to restore old recommendations and use them for the overall evaluation of the running experiment.

This is a sample config file with a proxy model restoring specific recommendations

.. code:: yaml

    experiment:
        ...
        model:
            external.ProxyRecommender:
            path: path/to/recs/of/specific/model.tsv
            ItemKNN:
                ...


``external.ProxyRecommender`` is a fake recommender model which is able to restore old recommendation and prepare all
inner data structures to support evaluation like Elliot expects