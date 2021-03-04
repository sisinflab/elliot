Configuration file
======================


.. code:: yaml

    experiment:
      data_config:
        strategy: dataset|fixed|hierarchy
        dataloader: KnowledgeChainsLoader|DataSetLoader
        dataset_path: ../data/{0}/trainingset.tsv
        root_folder: ../data/{0}/splitting/
        train_path: ../data/{0}/trainingset.tsv
        validation_path: ../data/{0}/trainingset.tsv
        test_path: ../data/{0}/testset.tsv

.. code:: yaml

        side_information:
            feature_data: ../data/{0}/original/features.npy
            map: ../data/{0}/map.tsv
            features: ../data/{0}/features.tsv
            properties: ../data/{0}/properties.conf


.. code:: yaml

      splitting:
        save_on_disk: True
        save_folder: ../data/{0}/splitting/
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

.. code:: yaml

      prefiltering:
        strategy: global_threshold|user_average|user_k_core|item_k_core|iterative_k_core|n_rounds_k_core|cold_users
        threshold: 3|average
        core: 5
        rounds: 2
      dataset: categorical_dbpedia_ml1m
    #  dataset: example

.. code:: yaml

      path_output_rec_result: ../results/{0}/recs/
      path_output_rec_weight: ../results/{0}/weights/
      path_output_rec_performance: ../results/{0}/performance/
      path_logger_config: ./config/logger_config.yml
      path_log_folder: ../log/

.. code:: yaml

      top_k: 50
      evaluation:
        cutoff: 10
        simple_metrics: [ nDCG, Precision, Recall, ItemCoverage, HR, MRR, MAP, F1, Gini, SEntropy, EFD, EPC, AUC, GAUC, LAUC, MAE, MSE, RMSE]
        relevance_threshold: 1
        paired_ttest: True
        complex_metrics:
        - metric: DSC
          beta: 2
        - metric: SRecall
          feature_data: ../data/categorical_dbpedia_ml1m/map.tsv

.. code:: yaml

      gpu: -1 # -1 is not use GPU

.. code:: yaml

      models:
        MostPop:
          meta:
            save_recs: True