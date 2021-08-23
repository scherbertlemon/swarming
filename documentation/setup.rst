.. _refsetup:

Setup
=====

This document assumes you have ``conda`` installed, but any other python installation will do.

1. Clone the model repository from https://github.com/scherbertlemon/swarming
2. Provide a python environment with ``python >= 3.8``, e.g. with ``conda``

    ::

        conda create -n swarm python=3.8

    .. admonition:: Remark for Windows

        On Windows, it is recommended to install ``numpy, scipy, pandas, jupyter, graphviz, python-graphviz`` and ``pyarrow`` with conda, e.g. into a separate environment:

        ::

            conda create -n swarm python=3.8 numpy scipy jupyter pandas pyarrow


3. Install the ``swarming`` package into your environment (e.g. use ``conda activate`` to activate it) from the cloned repository folder:

    ::

        pip install -e .

After that, ``swarming`` can be imported from any notebook or program running in your environment.