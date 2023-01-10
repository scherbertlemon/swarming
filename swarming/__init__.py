"""
Simulation package for **interacting particle systems**
-------------------------------------------------------

Calculates a system of ordinary differential equations in terms of positions
``X`` and velocities ``V``, subject to force depending on all ``X`` and ``V``,
using an explicit Runge-Kutta scheme of second order.

Furthermore, there is functionality to record system state and system
visualization. Initial conditions, force terms (right hand side, RHS),
parameters for RHS, can be replaced for every time step, to allow for change
of parameters during a running simulation, e.g. for an interactive app.
"""

from .model import *
from .doe import *