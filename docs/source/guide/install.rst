Install Elliot
======================

Elliot works with the following operating systems:

-  Linux
-  Windows 10
-  macOS X

Elliot requires Python version 3.6 or later.

Elliot requires tensorflow version 2.3.2 or later. If you want to use Elliot with GPU,
please ensure that CUDA or cudatoolkit version is 7.6 or later.
This requires NVIDIA driver version >= 10.1 (for Linux and Windows10).

Please refer to this `document <https://www.tensorflow.org/install/source#gpu>`__ for further
working configurations.

Install from source
~~~~~~~~~~~~~~~~~~~

CONDA
^^^^^

.. code:: bash

    git clone https://github.com//sisinflab/elliot.git && cd elliot
    conda create --name elliot_env python=3.8
    conda activate
    pip install --upgrade pip
    pip install -e . --verbose

VIRTUALENV
^^^^^^^^^^

.. code:: bash

    git clone https://github.com//sisinflab/elliot.git && cd elliot
    virtualenv -p /usr/bin/pyhton3.6 venv # your python location and version
    source venv/bin/activate
    pip install --upgrade pip
    pip install -e . --verbose