Prefiltering Data
======================

After data loading,Elliot provides data filtering operations through two possible strategies. The first strategy
implemented in the Prefiltering module isFilter-by-rating, which drops off a user-item interaction if the preference
score is smaller than a given threshold. It can be (i) a Numerical value, e.g.,3.5, (ii)a Distributional detail, e.g.,
global rating average value, or (iii) a user-based distributional (User Dist.) value, e.g., user’s average rat-ing value.
The second prefiltering strategy,𝑘-core, filters out users,items, or both, with less than𝑘recorded interactions. The 𝑘-core
strategy can proceed iteratively (Iterative𝑘-core) on both users and items until the 𝑘-core filtering condition is met,
i.e., all the users and items have at least𝑘recorded interaction. Since reaching such condition might be intractable,
Elliot allows specifying the maximum number of iterations (Iter-n-rounds). Finally, the Cold-Users
filtering feature allows retaining cold-users only.

Elliot provides several prefiltering strategies.
To enable Prefiltering operations, we can insert the corresponding block into our config file:

.. code:: yaml

    experiment:
      prefiltering:
        strategy: global_threshold|user_average|user_k_core|item_k_core|iterative_k_core|n_rounds_k_core|cold_users
        threshold: 3|average
        core: 5
        rounds: 2

In detail, Elliot provides eight main prefiltering approaches: ``global_threshold``,
``user_average``, ``user_k_core``, ``item_k_core``, ``iterative_k_core``, ``n_rounds_k_core``, ``cold_users``.

``global_threshold`` assumes a single system-wise threshold to filter out irrelevant transactions.
``global_threshold`` takes one mandatory parameter, ``threshold``.
``threshold`` takes, as values, a **float** (ratings >= threshold will be kept), or the string *average*. With average, the system computes the global mean of the rating values and filters out all the ratings below.

.. code:: yaml

    experiment:
      prefiltering:
        strategy: global_threshold
        threshold: 3

.. code:: yaml

    experiment:
      prefiltering:
        strategy: global_threshold
        threshold: average

``user_average`` has no parameters, and the system filters out the ratings below each user rating values mean.

.. code:: yaml

    experiment:
      prefiltering:
        strategy: user_average

``user_k_core`` filters out all the users with a number of transactions lower than the given k core.
It takes a parameter, ``core``, where the user passes an **int** corresponding to the desired value.

.. code:: yaml

    experiment:
      prefiltering:
        strategy: user_k_core
        core: 5

``item_k_core`` filters out all the items with a number of transactions lower than the given k core.
It takes a parameter, ``core``, where the user passes an **int** corresponding to the desired value.

.. code:: yaml

    experiment:
      prefiltering:
        strategy: item_k_core
        core: 5

``iterative_k_core`` runs iteratively user_k_core, and item_k_core until the dataset is no further modified.
It takes a parameter, ``core``, where the user passes an **int** corresponding to the desired value.

.. code:: yaml

    experiment:
      prefiltering:
        strategy: iterative_k_core
        core: 5

``n_rounds_k_core`` runs iteratively user_k_core, and item_k_core for a specified number of rounds.
It takes the first parameter, ``core``, where the user passes an **int** corresponding to the desired value.
It takes the second parameter, ``rounds``, where the user passes an **int** corresponding to the desired value.

.. code:: yaml

    experiment:
      prefiltering:
        strategy: n_rounds_k_core
        core: 5
        rounds: 2

``cold_users`` filters out all the users with a number of interactions higher than a given threshold.
It takes a parameter, ``threshold``, where the user passes an **int** corresponding to the desired value.

.. code:: yaml

    experiment:
      prefiltering:
        strategy: cold_users
        threshold: 3

