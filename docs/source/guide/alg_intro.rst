Summary of the Recommendation Algorithms
============================================

Elliot integrates, to date, 50 recommendation models partitioned into two sets. The first set includes 38 popular models implemented in at least two of frameworks reviewed in this work (i.e., adopting a framework-wise popularity notion).

.. py:module:: elliot.recommender

All the recommendation models inherit from a common abstract class:

.. autosummary::
    base_recommender_model.BaseRecommenderModel

The majority of the recommendation models uses a Mixin:

.. autosummary::
    recommender_utils_mixin.RecMixin

* **Adversarial Learning**

.. autosummary::

    adversarial.AMF.AMF.AMF
    adversarial.AMR.AMR.AMR

* **Algebric**

.. autosummary::

    algebric.slope_one.slope_one.SlopeOne

* **Autoencoders**

.. autosummary::

    autoencoders.dae.multi_dae.MultiDAE
    autoencoders.vae.multi_vae.MultiVAE

* **Content-Based**

.. autosummary::

    content_based.VSM.vector_space_model.VSM

* **Generative Adversarial Networks (GANs)**

.. autosummary::

    gan.IRGAN.irgan.IRGAN
    gan.CFGAN.cfgan.CFGAN

* **Graph-based**

.. autosummary::

    graph_based.lightgcn.LightGCN.LightGCN
    graph_based.ngcf.NGCF.NGCF

* **Knowledge-aware**

.. autosummary::

    knowledge_aware.kaHFM.ka_hfm.KaHFM
    knowledge_aware.kaHFM_batch.kahfm_batch.KaHFMBatch
    knowledge_aware.kahfm_embeddings.kahfm_embeddings.KaHFMEmbeddings

* **Latent Factor Models**

.. autosummary::

    latent_factor_models.BPRMF.BPRMF.BPRMF
    latent_factor_models.BPRMF_batch.BPRMF_batch.BPRMF_batch
    latent_factor_models.BPRSlim.bprslim.BPRSlim
    latent_factor_models.CML.CML.CML
    latent_factor_models.FFM.field_aware_factorization_machine.FFM
    latent_factor_models.FISM.FISM.FISM
    latent_factor_models.FM.factorization_machine.FM
    latent_factor_models.FunkSVD.funk_svd.FunkSVD
    latent_factor_models.LogisticMF.logistic_matrix_factorization.LogisticMatrixFactorization
    latent_factor_models.MF.matrix_factorization.MF
    latent_factor_models.NonNegMF.non_negative_matrix_factorization.NonNegMF
    latent_factor_models.PMF.probabilistic_matrix_factorization.PMF
    latent_factor_models.PureSVD.pure_svd.PureSVD
    latent_factor_models.Slim.slim.Slim
    latent_factor_models.SVDpp.svdpp.SVDpp
    latent_factor_models.WRMF.wrmf.WRMF

* **Artificial Neural Networks**

.. autosummary::

    neural.ConvMF.convolutional_matrix_factorization.ConvMF
    neural.ConvNeuMF.convolutional_neural_matrix_factorization.ConvNeuMF
    neural.DeepFM.deep_fm.DeepFM
    neural.DMF.deep_matrix_factorization.DMF
    neural.GeneralizedMF.generalized_matrix_factorization.GMF
    neural.ItemAutoRec.itemautorec.ItemAutoRec
    neural.NAIS.nais.NAIS
    neural.NeuMF.neural_matrix_factorization.NeuMF
    neural.NFM.neural_fm.NFM
    neural.NPR.neural_personalized_ranking.NPR
    neural.UserAutoRec.userautorec.UserAutoRec
    neural.WideAndDeep.wide_and_deep.WideAndDeep

* **Neighborhood-based Models**

.. autosummary::

    NN.item_knn.item_knn.ItemKNN
    NN.user_knn.user_knn.UserKNN
    NN.attribute_item_knn.attribute_item_knn.AttributeItemKNN
    NN.attribute_user_knn.attribute_user_knn.AttributeUserKNN

* **Unpersonalized Recommenders**

.. autosummary::

    unpersonalized.most_popular.most_popular.MostPop
    unpersonalized.random_recommender.Random.Random

* **Visual Models**

.. autosummary::

    visual_recommenders.ACF.ACF.ACF
    visual_recommenders.DeepStyle.DeepStyle.DeepStyle
    visual_recommenders.DVBPR.DVBPR.DVBPR
    visual_recommenders.VBPR.VBPR.VBPR
    visual_recommenders.VNPR.visual_neural_personalized_ranking.VNPR
    visual_recommenders.elliot.recommender.adversarial.AMR.AMR.AMR