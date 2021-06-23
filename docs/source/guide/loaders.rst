Loaders
======================

If needed, it is possible to code a custom loader. To create your own loader, Elliot provides the abstract class
``AbstractLoader`` into the file ``abstract_loader.py`` belonging to the package

``dataset``
    |___``modular_loaders``


The ``init`` method are devoted to capture all the configuration provided by the loader section in configuration file
(like file path, file name, specific value, ...).

Another mandatory method is ``filter``, providing the filtering between