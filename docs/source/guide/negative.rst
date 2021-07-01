Negative Sampling
======================

Some evaluation strategies needed a collection of items for each user to rank as negative items alongside positive ones.
Elliot provides two different strategies to cover this feature. The first one is named *fixed* because needed for a file
containing the negative items for each user. The file containing these negative items must have tab-separated-value
with this pattern for each line

``(user_id, item_test_id)   neg_item1   neg_item2 ....``

This is a part of config file to handle this kind of solution

.. code:: yaml

    experiment:
        negative_sampling:
            strategy: fixed
            files: [ path/to/file ]

Another possible solution to address this feature is to choose a *uniform random* strategy. With this approach the
configuration file must be report also the number of negative samples to take into account.

Be careful: to guarantee the reproducibility of the experiment, in this case Elliot store in the dataset folder the
computed negative items following the following schema for a generic tab-separated-value

``(user_id, item_test_id)   neg_item1   neg_item2 ....``

This is a part of config file to handle this kind of solution

.. code:: yaml

    experiment:
        negative_sampling:
            strategy: random
            num_items: 5