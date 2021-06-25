Early Stopping
======================

Elliot lets the practitioner to save training time providing **Early stopping** functionalities.
To ease the understanding of the fields, we adopted (and slightly extended) the same notation as Keras, and PyTorch.

* **monitor**: Quantity to be monitored (metric @ cutoff, or 'loss').
* **min_delta**: Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
* **min_delta**: Minimum relative change in the monitored quantity to qualify as an improvement, i.e. a relative change of less than rel_delta, will count as no improvement.
* **patience**: Number of epochs with no improvement after which training will be stopped. **WARNING**: here, we only monitor epochs in the **validation_rate** epochs. If **validation_rate** == 1, patience consider single epochs. Otherwise, **patience** will count **ONE** epoch for each **validation_rate** step.
* **verbose**: verbosity mode.
* **mode**: One of {"auto", "min", "max"}. In min mode, training will stop when the quantity monitored has stopped decreasing; in "max" mode it will stop when the quantity monitored has stopped increasing; in "auto" mode, the direction is automatically inferred from the name of the monitored quantity.
* **baseline**: Baseline value for the monitored quantity. Training will stop if the model doesn't show improvement over the baseline.

Here, you can find an example that shows all the possible options.

.. code:: yaml

    experiment:
      models:
        MF2020: # from Rendle's 2020 KDD paper
          meta:
            hyper_max_evals: 1
            hyper_opt_alg: tpe
            validation_rate: 1
            verbose: True
            save_recs: True
          epochs: 256
          factors: 192
          lr: 0.007
          reg: 0.01
          m: 10
          random_seed: 42
          early_stopping:
            patience: 10 # int
            monitor: HR@10|loss
            mode: min|max|auto
            verbose: True|False
            min_delta: 0.001 # float absolute variation
            rel_delta: 0.1 # float (ratio)
            baseline: 0.1 # float absolute value