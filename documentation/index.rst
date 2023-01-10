.. swarming documentation master file, created by
   sphinx-quickstart on Sun Aug 22 17:21:22 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Simulation package for interacting particle systems
===================================================

Calculates a system of ordinary differential equations in terms of positions
``X`` and velocities ``V``, subject to force depending on all ``X`` and ``V``,
using an explicit Runge-Kutta scheme of second order.

It is primarily designed to model interacting forces that result in behavior of a swarm.

Furthermore, there is functionality to record system state and system
visualization. Initial conditions, force terms (right hand side, RHS),
parameters for RHS, can be replaced for every time step, to allow for change
of parameters during a running simulation, e.g. for an interactive app.

* Go to :ref:`refsetup` to learn about installation steps
* Go to :ref:`refliterature` to learn more about the model used in this package
* ``notebooks/presentation.ipynb`` contains the model idea as well as an overview about how to use this package.
* A Bokeh app is distributed with this package, which launches the swarm simulation and lets you modify the model parameters while it is running. It can be launched from the repository folder with

    ::

        bokeh serve --show bokeh_app/swarm.py

.. toctree::
    :maxdepth: 2
    :caption: Module contents:

    setup
    literature
    swarming

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
