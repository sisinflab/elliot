Neighborhood-based Models
======================

Elliot integrates, to date, 50 recommendation models partitioned into two sets. The first set includes 38 popular models implemented in at least two of frameworks reviewed in this work (i.e., adopting a framework-wise popularity notion).


.. py:module:: elliot.recommender.NN


Summary
~~~~~~~~~~~~~~~~

.. autosummary::

    item_knn.item_knn.ItemKNN
    user_knn.user_knn.UserKNN
    attribute_item_knn.attribute_item_knn.AttributeItemKNN
    attribute_user_knn.attribute_user_knn.AttributeUserKNN



ItemKNN
~~~~~~~~~~~~~~~~

.. autoclass:: elliot.recommender.NN.item_knn.item_knn.ItemKNN
    :show-inheritance:

UserKNN
~~~~~~~~~~~~~~~~

.. autoclass:: elliot.recommender.NN.user_knn.user_knn.UserKNN
    :show-inheritance:

AttributeItemKNN
~~~~~~~~~~~~~~~~

.. autoclass:: elliot.recommender.NN.attribute_item_knn.attribute_item_knn.AttributeItemKNN
    :show-inheritance:

AttributeUserKNN
~~~~~~~~~~~~~~~~

.. autoclass:: elliot.recommender.NN.attribute_user_knn.attribute_user_knn.AttributeUserKNN
    :show-inheritance: