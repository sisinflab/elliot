Loaders
======================

Elliot lets you plug in custom side-information loaders: code that reads item/user
attributes, a knowledge graph, textual/visual embeddings, or anything else keyed by a
user/item id, and hands it to your recommender as one of three canonical payload
shapes (``EmbeddingPayload``/``TextPayload``/``GraphPayload``, defined in
``elliot.dataset.modular_loaders.formats``).

To write one, subclass ``AbstractLoader`` (``elliot/dataset/modular_loaders/abstract_loader.py``)
and register it with ``side_info_registry``. A minimal loader only needs two methods:

- ``discover()``: a *lightweight pre-load* pass -- cheap I/O, just enough to find out
  which of the base users/items your data source actually covers, then narrow
  ``self.users``/``self.items`` down to that domain (typically
  ``self.items = self.items & {... ids found on disk ...}``). It must never
  materialize the full feature payload -- that is ``load()``'s job.
- ``load()``: materialize the actual payload, once, as a ``dict`` of named payloads.

``get_mapped()`` and ``filter()`` already have sensible defaults on ``AbstractLoader``
and rarely need overriding (see `When to override filter()`_ below).

Reading the raw file
---------------------

Don't hand-roll file parsing inside your loader. ``self.reader`` (an
``elliot.utils.read.Reader``, available on every loader) already covers most on-disk
layouts Elliot's side-information files come in:

- ``read_mapping``: an ``id -> [value, ...]`` file (e.g. multi-hot feature ids per item).
- ``read_id_mapping``: an ``id -> single value`` file (e.g. item id -> KG entity URI).
- ``read_property_list``: a newline-delimited list (e.g. predicate URIs), skipping
  ``#``-comment lines.
- ``read_json`` / ``read_triples``: JSON files / headerless ``(subject, predicate,
  object)`` KG triples.
- ``read_folder`` / ``peek_npy_shape`` / ``read_npy``: listing a folder, cheaply
  sniffing a ``.npy`` array's shape (via a memory-mapped read) without loading its
  data, and fully loading one when you do need it.
- ``discover_npy_ids``: the common "one ``.npy`` file per id" folder layout in one
  call -- lists the folder, extracts ids from filenames, and shape-sniffs the first
  file, returning ``(ids, id_map, shape)``.
- ``read_triples_as_tuples``: KG triples as `List[(subject, predicate, object)]`.
- ``read_tabular`` / ``read_sequence_tabular``: general-purpose tabular readers, for
  layouts none of the above fit.

Reader only ever returns raw, already-parsed data (dicts, DataFrames, arrays, id
sets) -- never one of the canonical payload dataclasses. Shaping that raw data into a
payload is what the adapters in ``elliot.dataset.modular_loaders.adapters`` do (see
below); keeping the two responsibilities apart is what lets both be reused
independently across loaders.

If a raw layout genuinely isn't covered yet, add a method to ``Reader`` for it rather
than parsing it by hand inside the loader -- that keeps every loader focused on *what*
it loads, not *how* to read a file.

For raw-format -> canonical-payload conversions (a folder of one ``.npy`` file per id, a
categorical feature map, a pairwise similarity JSON, RDF triples, ...), check
``elliot.dataset.modular_loaders.adapters`` first too. Most existing loaders are just
"read something small with a ``Reader``/adapter helper, then hand it to another adapter
helper" -- no bespoke parsing at all.

Example: item attributes
-------------------------

Suppose we want to load per-item categorical attributes (e.g. movie genres) from a
headerless TSV file, one row per item: the first column is the item id, the remaining
columns are genre ids.

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

This is exactly what ``elliot.dataset.modular_loaders.generic.item_attributes.ItemAttributes``
does:

.. code:: python

    @side_info_registry.register(
        provides="item_features",
        format="embedding",
        alignment=AlignmentMode.DROP,
    )
    class ItemAttributes(AbstractLoader):
        attribute_file: str

        def discover(self):
            self.map_ = self.reader.read_mapping(self.attribute_file, dtype="int")
            self.items = self.items & set(self.map_.keys())

        def load(self):
            return {"item_features": raw_feature_map_to_embedding_payload(self.map_, self.items)}

That's the whole loader: ``get_mapped()``/``filter()`` are inherited from
``AbstractLoader`` unchanged, the file is read with ``self.reader.read_mapping``
(no hand-rolled ``open()``/``split()``), and the multi-hot -> ``EmbeddingPayload``
conversion is delegated to ``raw_feature_map_to_embedding_payload`` (in
``elliot.dataset.modular_loaders.adapters``).

Configuration
--------------

Any field declared as a class attribute on your loader (``attribute_file: str`` above)
is populated automatically from the matching key in the ``side_information`` block of
the experiment configuration:

.. code:: yaml

    experiment:
      data_config:
        side_information:
          - dataloader: ItemAttributes
            attribute_file: this/is/the/path.tsv

The ``dataloader`` name must match the class name, or the ``name=`` given to
``side_info_registry.register(...)``.

When to override ``filter()``
------------------------------

``filter()`` is called once, globally, to narrow every loader down to the final
cross-loader user/item intersection. The default on ``AbstractLoader`` just intersects
``self.users``/``self.items``, which is correct for any loader that keeps no other
id-keyed state.

Override it -- calling ``super().filter(users, items)`` first -- only when some other
attribute set up in ``discover()`` (a raw triples table, a derived feature map, ...)
also needs to be narrowed down to stay consistent with the new
``self.users``/``self.items``. See
``elliot.dataset.modular_loaders.kg.kahfm_style.ChainedKG`` for an example that
re-derives its feature map after filtering.
