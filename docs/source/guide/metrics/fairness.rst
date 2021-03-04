Fairness
======================

Elliot integrates, to date, 50 recommendation models partitioned into two sets. The first set includes 38 popular models implemented in at least two of frameworks reviewed in this work (i.e., adopting a framework-wise popularity notion).

Summary
~~~~~~~~~~~~~~~~

.. py:module:: elliot.evaluation.metrics.fairness
.. autosummary::
    BiasDisparity.BiasDisparityBD.BD
    BiasDisparity.BiasDisparityBR.BR
    BiasDisparity.BiasDisparityBS.BS
    MAD.ItemMADranking.ItemMADranking
    MAD.ItemMADrating.ItemMADrating
    MAD.UserMADranking.UserMADranking
    MAD.UserMADrating.UserMADrating
    reo.reo.REO
    rsp.rsp.RSP

BiasDisparity BD
~~~~~~~~~~~~~~~~
.. module:: elliot.evaluation.metrics.fairness.BiasDisparity.BiasDisparityBD
.. autoclass:: BiasDisparityBD
    :show-inheritance:

BiasDisparity BR
~~~~~~~~~~~~~~~~
.. module:: elliot.evaluation.metrics.fairness.BiasDisparity.BiasDisparityBR
.. autoclass:: BiasDisparityBR
    :show-inheritance:

BiasDisparity BS
~~~~~~~~~~~~~~~~
.. module:: elliot.evaluation.metrics.fairness.BiasDisparity.BiasDisparityBS
.. autoclass:: BiasDisparityBS
    :show-inheritance:

ItemMADranking
~~~~~~~~~~~~~~~~
.. module:: elliot.evaluation.metrics.fairness.MAD.ItemMADranking
.. autoclass:: ItemMADranking
    :show-inheritance:

ItemMADrating
~~~~~~~~~~~~~~~~
.. module:: elliot.evaluation.metrics.fairness.MAD.ItemMADrating
.. autoclass:: ItemMADrating
    :show-inheritance:

UserMADranking
~~~~~~~~~~~~~~~~
.. module:: elliot.evaluation.metrics.fairness.MAD.UserMADranking
.. autoclass:: UserMADranking
    :show-inheritance:

UserMADrating
~~~~~~~~~~~~~~~~
.. module:: elliot.evaluation.metrics.fairness.MAD.UserMADrating
.. autoclass:: UserMADrating
    :show-inheritance:

REO
~~~~~~~~~~~~~~~~
.. module:: elliot.evaluation.metrics.fairness.reo.reo
.. autoclass:: REO
    :show-inheritance:

RSP
~~~~~~~~~~~~~~~~
.. autoclass:: elliot.evaluation.metrics.fairness.rsp.rsp.RSP
    :show-inheritance:
