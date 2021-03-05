Splitting Data
======================

If needed, the data is served to the Splitting module. In detail, Elliot provides (i)Temporal, (ii)Random, and (iii)Fix strategies.
The Temporal strategy splits the user-item interactions based on the transaction timestamp, i.e., fixing the timestamp, find-ing the optimal one, or adopting a hold-out (HO) mechanism.
The Random strategy includes hold-out (HO),𝐾-repeated hold-out(K-HO), and cross-validation (CV). Table 1 provides further
configuration details. Finally, the Fix strategy exploits a precomputed splitting.

Elliot provides several splitting strategies.
To enable the splitting operations, we can insert the corresponding section:

.. code:: yaml

    experiment:
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

    experiment:
      splitting:
        save_on_disk: True
        save_folder: this/is/the/path/

Now, we can insert one (or two) specific subsections to detail the train/test, and the train/validation splitting via the corresponding fields:
``test_splitting``, and ``validation_splitting``.
``test_splitting`` is clearly mandatory, while ``validation_splitting`` is optional.
Since the two subsections follow the same guidelines, here we detail ``test_splitting`` without loss of generality.

Elliot enables four splitting families: ``fixed_timestamp``, ``temporal_hold_out``, ``random_subsampling``, ``random_cross_validation``.

``fixed_timestamp`` assumes that there will be a specific timestamp to split prior interactions (train) and future interactions.
It takes the parameter ``timestamp``, that can assume one of two possible kind of values: a **long** corresponding to a specific timestamp, or the string *best* computed following `Anelli et al. <https://doi.org/10.1007/978-3-030-15712-8_63>`_

.. code:: yaml

    experiment:
      splitting:
        test_splitting:
            strategy: fixed_timestamp
            timestamp: 1609786061

.. code:: yaml

    experiment:
      splitting:
        test_splitting:
            strategy: fixed_timestamp
            timestamp: best

``temporal_hold_out`` relies on a temporal split of user transactions. The split can be realized following two different approaches: a *ratio-based* and a *leave-n-out-based* approach.
If we enable the ``test_ratio`` field with a **float** value, Elliot splits data retaining the last (100 * ``test_ratio``) % of the user transactions for the test set.
If we enable the ``leave_n_out`` field with an **int** value, Elliot retains the last ``leave_n_out`` transactions for the test set.

.. code:: yaml

    experiment:
      splitting:
        test_splitting:
            strategy: temporal_hold_out
            test_ratio: 0.2

.. code:: yaml

    experiment:
      splitting:
        test_splitting:
            strategy: temporal_hold_out
            leave_n_out: 1

``random_subsampling`` generalizes random hold-out strategy.
It takes a ``test_ratio`` parameter with a **float** value to define the train/test ratio for user-based hold-out splitting.
Alternatively, it can take ``leave_n_out`` with an **int** value to define the number of transaction retained for the test set.
Moreover, the splitting operation can be repeated enabling the ``folds`` field and passing an **int**.
In that case, the overall splitting strategy corresponds to a user-based random subsampling strategy.

.. code:: yaml

    experiment:
      splitting:
        test_splitting:
            strategy: random_subsampling
            test_ratio: 0.2

.. code:: yaml

    experiment:
      splitting:
        test_splitting:
            strategy: random_subsampling
            test_ratio: 0.2
            folds: 5

.. code:: yaml

    experiment:
      splitting:
        test_splitting:
            strategy: random_subsampling
            leave_n_out: 1
            folds: 5

``random_cross_validation`` adopts a k-folds cross-validation splitting strategy.
It takes the parameter ``folds`` with an **int** value, that defines the overall number of folds to consider.

.. code:: yaml

    experiment:
      splitting:
        test_splitting:
            strategy: random_cross_validation
            folds: 5