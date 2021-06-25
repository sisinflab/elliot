Loaders
======================

If needed, it is possible to code a custom loader. To create your own loader, Elliot provides the abstract class
``AbstractLoader`` into the file ``abstract_loader.py`` belonging to the package

``dataset``
    |___``modular_loaders``


The ``init`` method is devoted to capture all the configuration fields provided by the loader section in configuration file and initialize the loader (read files, apply thresholds, create additional data structures).

Example: Suppose we want to create a side information loader for movie genres. The side information file is structured as a TSV file with no header.
The first element of the row denotes the item id, whereas the other numbers indicate the ids of the genres.

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - item id
     - genre0
     - genre1
     - genreN
   * - 1
     - 0
     - 1
     - 5
   * - 2
     - 7
     - 0
     -
   * - 3
     - 2
     - 3
     -


.. code:: python

    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.attribute_file = getattr(ns, "attribute_file", None)
        self.users = users
        self.items = items
        self.map_ = self.load_attribute_file(self.attribute_file)
        self.items = self.items & set(self.map_.keys())

The ``__init__`` (mandatory) method takes four mandatory arguments: users, items, the namespace, and the elliot general logger.
In our example, the namespace corresponds to the piece of the configuration file that refers to our side information loader (form now on named ItemAttributes).

.. code:: yaml

    experiment:
      data_config:
        side_information:
          - dataloader: ItemAttributes
            attribute_file: this/is/the/path.tsv

The ``__init__`` method creates its local attributes and retrieve the necessary information from the namespace.
Then, it loads the side information file and aligns it with users and items as provided by the Elliot pipeline.

The method ``get_mapped()`` (mandatory), returns a tuple of aligned users and items.

.. code:: python

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

The method ``filter`` (mandatory), provides the functionality of filtering users, items, and side information data structures based on the sets of users and items passed as arguments.

.. code:: python

    def filter(self, users, items):
        self.users = self.users & users
        self.items = self.items & items

Finally, the method ``create_namespace`` creates the namespace that will be passed to our recommendation algorithms.
Be sure that the mandatory attributes (__name__, and object), and all the necessary data are present.
Pay Attention! The name you choose here is the same you will use in your configuration file.

.. code:: python

    def create_namespace(self):
        ns = SimpleNamespace()
        ns.__name__ = "ItemAttributes" #MANDATORY
        ns.object = self #MANDATORY
        ns.feature_map = self.map_
        ns.features = list({f for i in self.items for f in ns.feature_map[i]})
        ns.nfeatures = len(ns.features)
        ns.private_features = {p: f for p, f in enumerate(ns.features)}
        ns.public_features = {v: k for k, v in ns.private_features.items()}
        return ns