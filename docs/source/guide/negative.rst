Evaluation Protocol
======================

The evaluation protocol answers two independent questions, both configured in the ``evaluation.protocol`` block:

1. **What does each evaluation row predict, given what?** → ``target``
2. **Which candidates is each row ranked against?** → ``negative_sampling`` (absent: the whole catalog)

The first one also depends on how models read their input, i.e. on their ``meta.session_strategy``:

- **non-sequential** models (e.g. BPRMF, ItemKNN) and **FLAT** sequential models evaluate one sequence per user:
  its known history followed by its eval items. Non-sequential models only use their trained user
  representation; FLAT sequential ones also read the history and the eval items preceding the target;
- **SESSION_ONLY** sequential models evaluate one sequence per eval session, whose context is only the
  session prefix (the first item of a session is never a target). They need a dataset segmented into
  sessions (``data_config.session_strategy: session_only``).

Evaluation target
------------------

A single ``target`` applies to every model of the experiment:

.. code:: yaml

    experiment:
      evaluation:
        protocol:
          target: all|first|last|random|successive

- ``all``: one row per user, predicting all of its eval items from its history (the classic protocol);
- ``first``: one row per user, predicting its first eval item from its history;
- ``last``: one row per sequence, predicting its last item from everything preceding it;
- ``random``: one row per sequence, predicting a random (seeded) item from everything preceding it;
- ``successive``: one row per eval item, predicting it from everything preceding it; rows are averaged per user.

When ``target`` is not set, it is ``last`` if any model is SESSION_ONLY, ``all`` otherwise.
Validation always uses ``last``, unless ``target`` is ``all``.
Any target but ``all`` needs a temporal splitting.

Which targets are available depends on the models of the experiment, so that every model is scored on the
same rows and results stay comparable:

======================================================  ==========================================
Models in the experiment                                Available targets
======================================================  ==========================================
non-sequential and/or FLAT only                         ``all``, ``first``, ``last``, ``random``, ``successive``
SESSION_ONLY only                                       ``last``, ``random``, ``successive``
SESSION_ONLY mixed with non-sequential and/or FLAT      ``last`` only, with ``test_splitting`` set to
                                                        ``temporal_holdout`` and ``leave_n_out: 1``
======================================================  ==========================================

In a mixed experiment the test set holds each user's last session (``leave_n_out`` counts sessions on
session-segmented data), and FLAT rows whose target opens a session (e.g. single-item test sessions) are
dropped, since SESSION_ONLY models can never score them: both kinds of models then predict the very same
items for the very same users, each from the context it can read.

Negative sampling
------------------

Without ``negative_sampling``, each row is ranked against the whole catalog. With it, each row is ranked
against its target item(s) plus its user's sampled negatives. Negatives are sampled once per user and fold,
and shared by every row of that user, whatever the model, so all models face the same candidates.
Negatives never include items the user interacted with (train, validation history, or eval).

Negative sampling only decides the candidates: which items are predicted is decided by ``target`` alone
(e.g. ``target: first`` to predict only the first eval item).

The *random* strategy samples ``num_negatives`` items uniformly at random for each user, and can store
them on disk for reproducibility:

.. code:: yaml

    experiment:
      evaluation:
        protocol:
          negative_sampling:
            strategy: random
            num_negatives: 99
            save_on_disk: True
            save_folder: path/to/folder

The *fixed* strategy reads them from tab-separated files in ``read_folder``, with the following pattern
for each line:

``user_id   neg_item1   neg_item2 ....``

.. code:: yaml

    experiment:
      evaluation:
        protocol:
          negative_sampling:
            strategy: fixed
            read_folder: path/to/folder
