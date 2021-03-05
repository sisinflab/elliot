## Basic Configuration

In the first scenario, the experiments require comparing a group of RSs whose parameters are optimized via a grid-search. 

The configuration specifies the data loading information, i.e., semantic features source files, in addition to the filtering and splitting strategies. 

In particular, the latter supplies an entirely automated way of preprocessing the dataset, which is often a time-consuming 
and non-easily-reproducible phase. 

The simple_metrics field allows computing accuracy and beyond-accuracy metrics, with two top-k cut-off values (5 and 10) 
by merely inserting the list of desired measures, e.g., [Precision, nDCG, ...]. 
The knowledge-aware recommendation model, AttributeItemKNN, is compared against two baselines: Random and ItemKNN, 
along with a user-implemented model that is external.MostPop. 

The configuration makes use of elliot's feature of conducting a grid search-based hyperparameter optimization strategy
by merely passing a list of possible hyperparameter values, e.g., neighbors: [50, 70, 100]. 

The reported models are selected according to nDCG@10.

|To see the full configuration file please visit the following [link](config_files/basic_configuration.yml)|
|-------------------------------------------------------------------------------------------------------------|
|**To run the experiment use the following [script](sample_basic.py)**|