This configuration file [link] takes movielens dataset from a specific path, then Elliot performs an exhaustive iterative k-core for both user 
and item with a minimum number of 10 interactions. Later, a splitting strategy with test and validation solutions is adopted. 
The test is split with a random subsampling for 1 fold and with a ratio of 20% with respect to the amount of data. Instead, 
the validation portion is computed in cross-validation with 5 folds. In this way, models will declare in the following section are 
trained 5 times (once per each train-validation pair) to estimate the validation performance.

The next section of this configuration file is devoted to declaring the evaluation metrics and which cut-off Elliot has 
to investigate to perform this evaluation step. The framework accepts both simple metrics (metrics that do not exploit 
external files o configurations) and complex metrics (like metrics related to bias o fairness investigation). Note that 
Elliot has a top_k parameter useful to produce recommendation lists with a specific number of relevant items, and for 
the evaluation could have specific cut-offs.

The third part of this YAML structured file declares explicitly which models Elliot has to train and evaluate. 
This section is the most expressive one because each model could be equipped with a specific hyperparameter exploration strategy. 
Specifically, this file shows how NeuMF and MultiVae adopt a Bayesian optimization exploration named TPE (Tree Parzen Estimator), 
which extracts 5 different model configurations that exploit the space strategy adopted by different parameters in both models.[to be continued]

To see the full configuration file please visit the following [link](config_files/advanced_configuration.yml).
To run the experiment use the following [script](sample_advanced.py).


```yaml
experiment:
  dataset: movielens_1m
  data_config:
    strategy: dataset
    dataset_path: ../data/movielens_1m/dataset.tsv
  prefiltering:
    strategy: iterative_k_core
    core: 10
  splitting:
    save_folder: ../data/movielens_1m/splitting/
    test_splitting:
        strategy: random_subsampling
	folds: 1
        test_ratio: 0.2
    validation_splitting:
        strategy: random_cross_validation
        folds: 5
  top_k: 50
  evaluation:
    cutoff: 10
    simple_metrics: [nDCG, ACLT, APLT, ARP, PopREO]
    complex_metrics: 
    - metric: UserMADrating
      clustering_name: Happiness
      clustering_file: ../data/movielens_1m/u_happy.tsv
    - metric: ItemMADrating
      clustering_name: ItemPopularity
      clustering_file: ../data/movielens_1m/i_pop.tsv
    - metric: REO
      clustering_name: ItemPopularity
      clustering_file: ../data/movielens_1m/i_pop.tsv
    - metric: RSP
      clustering_name: ItemPopularity
      clustering_file: ../data/movielens_1m/i_pop.tsv
    - metric: BiasDisparityBD
      user_clustering_name: Happiness
      user_clustering_file: ../data/movielens_1m/u_happy.tsv
      item_clustering_name: ItemPopularity
      item_clustering_file: ../data/movielens_1m/i_pop.tsv
    relevance_threshold: 1
  gpu: 1
  models:
    NeuMF:
      meta:
        hyper_max_evals: 5
        hyper_opt_alg: tpe
        validation_rate: 5
      lr: [loguniform, -10, -1]
      batch_size: [128, 256, 512]
      epochs: 50
      mf_factors: [quniform, 8, 32, 1]
      mlp_factors: [8, 16]
      mlp_hidden_size: [(32, 16, 8), (64, 32, 16)]
      prob_keep_dropout: 0.2
      is_mf_train: True
      is_mlp_train: True
    MultiVAE:
      meta:
        hyper_max_evals: 5
        hyper_opt_alg: tpe
        validation_rate: 5
      lr: [0.0005, 0.001, 0.005, 0.01]
      epochs: 50
      batch_size: [128, 256, 512]
      intermediate_dim: [300, 400, 500]
      latent_dim: [100, 200, 300]
      dropout_pkeep: 1
      reg_lambda: [0.1, 0.0, 10]
    BPRMF:
      meta:
        hyper_max_evals: 5
        hyper_opt_alg: rand
        validation_rate: 5
      lr: [0.0005, 0.001, 0.005, 0.01]
      batch_size: [128, 256, 512]
      epochs: 50
      embed_k: [10, 50, 100]
      bias_regularization: 0
      user_regularization: [0.0025, 0.005, 0.01]
      positive_item_regularization: [0.0025, 0.005, 0.01]
      negative_item_regularization: [0.00025, 0.0005, 0.001]
      update_negative_item_factors: True
      update_users: True
      update_items: True
      update_bias: True
```