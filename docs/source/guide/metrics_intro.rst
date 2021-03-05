Metrics
======================

Elliot provides 36 evaluation metrics, partitioned into seven families: Accuracy, Rating-Error, Coverage, Novelty, Diversity, Bias, and Fairness.
It is worth mentioning that Elliot is the framework that exposes both the largest number of metrics and the only one considering bias and fairness measures.
Moreover, the user can choose any metric to drive the model selection and the tuning.

.. py:module:: elliot.evaluation.metrics

All the metrics inherit from a common abstract class:

.. autosummary::
    base_metric.BaseMetric

* **Accuracy**

.. autosummary::

    accuracy.AUC.auc.AUC
    accuracy.AUC.gauc.GAUC
    accuracy.AUC.lauc.LAUC
    accuracy.DSC.dsc.DSC
    accuracy.f1.f1.F1
    accuracy.f1.extended_f1.ExtendedF1
    accuracy.hit_rate.hit_rate.HR
    accuracy.map.map.MAP
    accuracy.mar.mar.MAR
    accuracy.mrr.mrr.MRR
    accuracy.ndcg.ndcg.NDCG

* **Bias**

.. autosummary::

    bias.aclt.aclt.ACLT
    bias.aplt.aplt.APLT
    bias.arp.arp.ARP
    bias.pop_reo.pop_reo.PopREO
    bias.pop_reo.extended_pop_reo.ExtendedPopREO
    bias.pop_rsp.pop_rsp.PopRSP
    bias.pop_rsp.extended_pop_rsp.ExtendedPopRSP

* **Coverage**

.. autosummary::

    coverage.item_coverage.item_coverage.ItemCoverage
    coverage.num_retrieved.num_retrieved.NumRetrieved
    coverage.user_coverage.user_coverage.UserCoverage
    coverage.user_coverage.user_coverage_at_n.UserCoverageAtN

* **Diversity**

.. autosummary::

    diversity.gini_index.gini_index.GiniIndex
    diversity.shannon_entropy.shannon_entropy.ShannonEntropy
    diversity.SRecall.srecall.SRecall

* **Fairness**

.. autosummary::

    fairness.BiasDisparity.BiasDisparityBD
    fairness.BiasDisparity.BiasDisparityBR
    fairness.BiasDisparity.BiasDisparityBS
    fairness.MAD.ItemMADranking.ItemMADranking
    fairness.MAD.ItemMADrating.ItemMADrating
    fairness.MAD.UserMADranking.UserMADranking
    fairness.MAD.UserMADrating.UserMADrating
    fairness.reo.reo.REO
    fairness.rsp.rsp.RSP

* **Novelty**

.. autosummary::

    novelty.EFD.efd.EFD
    novelty.EFD.extended_efd.ExtendedEFD
    novelty.EPC.epc.EPC
    novelty.EPC.extended_epc.ExtendedEPC

* **Rating**

.. autosummary::

    rating.mae.mae.MAE
    rating.mse.mse.MSE
    rating.rmse.rmse.RMSE