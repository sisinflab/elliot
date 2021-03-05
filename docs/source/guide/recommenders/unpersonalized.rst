Unpersonalized Recommenders
==============================

.. py:module:: elliot.recommender.unpersonalized


Summary
~~~~~~~~~~~~~~~~

.. autosummary::

    most_popular.most_popular.MostPop
    random_recommender.Random.Random

Most Popular
~~~~~~~~~~~~~~~~
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed id porta mi. Proin luctus sapien ut mauris facilisis, in faucibus quam cursus. Pellentesque eget lacus eros. Aenean eget molestie magna. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Nam dapibus erat at scelerisque facilisis. Cras diam dolor, viverra et ipsum ac, ultrices lacinia eros.

.. module:: elliot.recommender.unpersonalized.most_popular.most_popular
.. autoclass:: MostPop
    :show-inheritance:

To include the recommendation model, add it to the config file adopting the following pattern:
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code:: yaml

  models:
    MostPop:
      meta:
        save_recs: True


Random Recommender
~~~~~~~~~~~~~~~~~~~~
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed id porta mi. Proin luctus sapien ut mauris facilisis, in faucibus quam cursus. Pellentesque eget lacus eros. Aenean eget molestie magna. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Nam dapibus erat at scelerisque facilisis. Cras diam dolor, viverra et ipsum ac, ultrices lacinia eros.

.. module:: elliot.recommender.unpersonalized.random_recommender.Random
.. autoclass:: Random
    :show-inheritance:

To include the recommendation model, add it to the config file adopting the following pattern:
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code:: yaml

  models:
    Random:
      meta:
        save_recs: True
      random_seed: 42