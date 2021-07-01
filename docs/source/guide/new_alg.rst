Create a new Recommendation Model
======================

Elliot is design as platform to fairly compare a lot of state-of-the-art models and over, belonging to different families
of recommender systems. Obviously, someone could implement a new method and test it in our framework.

To create a new model and enable it on Elliot framework follow these steps:

1. create a python package into the models package placed into external folder
2. into this new package crate the python file containing the principle class for the model. This class must extend the mixin class ``RecMixin`` and the abstract class ``BaseRecommenderModel``
3. create the ``__init__`` method and annotated it with ``@init_charger``
4. the *init* method have to set up the parameters list coming from configuration and build it calling ``self.autoset_params()``
5. parameter list must follow this schema:

.. code:: python

    self._params_list = [
        (local_variable_name, string_from_config, short_name, default_value, casting_type, transform_function),
        ......
    ]

6. instantiate the variable model containing the recommender approach to match user's preferences
7. define your training strategy into the method ``train``
8. define, eventually, a custom strategy to compute the recommendations lists in order to evaluate them. Specifically, two methods needed: ``get_recommendations`` to prepare all predictions and ``get_single_recommendation`` to generate ranked list for each user
