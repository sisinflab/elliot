## Advanced Configuration

The second scenario depicts a more complex experimental setting. 
In the configuration, the user specifies an elaborate data splitting strategy, i.e., random_subsampling (for test splitting) 
and random_cross_validation (for model selection), by setting few splitting configuration fields. 

The configuration does not provide a cut-off value, and thus a top-k field value of 50 is assumed as the cut-off. 

Moreover, the evaluation section includes the UserMADrating metric.

Elliot considers it as a complex metric since it requires additional arguments.

The user also wants to implement a more advanced hyperparameter tuning optimization. For instance, regarding NeuMF, 
Bayesian optimization using Tree of Parzen Estimators is required (i.e., hyper_opt_alg: tpe) with a logarithmic uniform 
sampling for the learning rate search space.

Moreover, Elliot allows considering complex neural architecture search spaces by inserting lists of tuples. For instance, 
(32, 16, 8) indicates that the neural network consists of three hidden layers with 32, 16, and 8 units, respectively.


|To see the full configuration file please visit the following [link](config_files/advanced_configuration.yml)|
|-------------------------------------------------------------------------------------------------------------|
|**To run the experiment use the following [script](sample_advanced.py)**|
