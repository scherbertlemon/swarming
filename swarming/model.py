"""
Module containing most of the computation logic and the visualizations, using
Bokeh and ``ColumnDataSource`` to allow for updating figures displaying the
system.
"""

import numpy as np
from numpy.random import rand, randn
from bokeh.plotting import ColumnDataSource, figure
from bokeh.layouts import grid
from bokeh.models import ColorBar, LinearColorMapper
import pandas as pd


def force_of_distance(r, Ca=20.0, Cr=50.0, la=100, lr=2):
    """
    Force function for swarm behavior for illustration purposes

    Parameters
    ----------
    r: float, array
        vector of distances for which to compute forces
    Ca: float
        attraction force
    Cr: float
        repulsion force
    la: float
        attraction range
    lr: float
        repulsion range

    Returns
    -------
    numpy.array
        array of forces the same size as r
    """
    return Ca/la*np.exp(-r/la)-Cr/lr*np.exp(-r/lr)


def harmonic_oscillator_rhs(const=0.01, damping=0.1, **kwargs):
    """
    Right hand side for particle system, modelling harmonic oscillator behavior.

    Parameters
    ----------
    const: float
        coefficient
    damping: float
        damping coefficient
    
    Returns
    -------
    function
        the parameterized function to act as a right hand side for the system
    """
    def apply_rhs(X, V):
        """
        Parameterized function to apply on positions ``X`` and velocities ``V`` of the particle system.
        """
        return V, -const * X - damping * V

    return apply_rhs


def rep_attr_rhs(alpha=0.07, beta=0.05, Ca=20.0, Cr=50.0, la=100, lr=2, noise=0.0, **kwargs):
    """
    Right hand side for particle system, modelling swarming behavior by balancing a repulsive and attractive force.

    Parameters
    ----------
    alpha: float
        propulsion coefficient
    beta: float
        breaking/damping coefficient
    Ca: float
        attraction force
    Cr: float
        repulsion force
    la: float
        attraction range
    lr: float
        repulsion range
    noise: float
        coefficient for random noise added to the motion of the system
    
    Returns
    -------
    function
        the parameterized function to act as a right hand side for the system
    """
    def apply_rhs(X, V):
        """
        Parameterized function to apply on positions ``X`` and velocities ``V`` of the particle system.
        """
        def force(r):
            return Ca/la*np.exp(-r/la)-Cr/lr*np.exp(-r/lr)

        x1, x2 = np.meshgrid(X[:, 0], X[:, 0])
        distx1 = x1 - x2
        x1, x2 = np.meshgrid(X[:, 1], X[:, 1])
        distx2 = x1 - x2
        normdi = np.sqrt(distx1**2 + distx2**2)
        
        rhsv = V*(alpha - beta * (V**2 + np.flip(V, axis=1)**2)) - np.stack((
            np.sum(force(normdi)/((np.abs(normdi-0)<1e-14).astype(int) +normdi)*distx1, axis=0),
            np.sum(force(normdi)/((np.abs(normdi-0)<1e-14).astype(int) +normdi)*distx2, axis=0)
        )).T/X.shape[0] - 0.5 * noise**2 * V
        
        return V, rhsv
    
    return apply_rhs


class Model:
    """
    Base class for simulating a system of potentially interacting particles,
    as a system of ordinary differential equations in terms of particle
    positions ``X`` and velocities ``V``. The system behavior is mainly
    determined by the choice of the right hand side, which can be
    parameterized with coefficients, and which is applied to positions and
    velocities in every time step. There is further functionality to plot
    current model states with bokeh, and to record the complete trajectories
    of particles in the phase space.

    Parameters
    ----------
    X: numpy.array
        ``n`` by 2 array containing in each "row" the 2d positions of one of
        the ``n`` particles to simulate. Can be an initial distribution or an
        intermediate state. See also
        :py:class:`swarming.model.InitialCondition`
    V: numpy.array
        ``n`` by 2 array containing in each "row" the 2d velocity vector of
        one of the ``n`` particles to simulate.
    rhs: function
        function returning a reference to a parameterized right hand side to
        be evaluated in every timestep for positions and velocities.

    Attributes
    ----------
    X: numpy.array
        ``n`` by 2 array containing in each "row" the current 2d positions of
        one of the ``n`` particles in the system
    V: numpy.array
        ``n`` by 2 array containing in each "row" the current 2d velocity
        vectors of one of the ``n`` particles in the system
    time: float
        current simulation time until which the model has been run since
        instantiation.
    rhs: function
        function returning a reference to a parameterized right hand side to
        be evaluated in every timestep for positions and velocities. Can be
        changed during system evolution to modify parameters
    apply_rhs: function
        the return of ``rhs`` called with a set of parameters, to be applied
        to particles and velocities in the simulation
    VSCALE: float
        factor by which velocity vecotrs are stretched for visualization only.
    mean_x: numpy.array
        2 array always containing the mean position of all particles at
        current timestamp.
    mean_v: numpy.array
        2 array always containing the mean velocity of all particles at
        current timestamp.
    cds: bokeh.ColumnDataSource
        reference to column data source that is used for plotting system state.
        Can be updated with current simulated data by calling
        :py:meth:`swarming.model.Model.update_cds`. Helpful for updating bokeh
        plots that use this data source, e.g. in an interactive Bokeh app.
    history: pandas.DataFrame
        contains particle positions, velocities, mean position and velocity of
        all particles at certain time steps, that are determined in
        :py:meth:`swarming.model.record_for_time` or
        :py:meth:`swarming.model.record_until`.
    n_particles: int
        amount of particles, i.e. number of rows of ``X``: ``self.X.shape[0]``
    """
    def __init__(self, X, V, rhs=None):
        self.X = X
        self.V = V
        self.VSCALE = 1.5
        if rhs:
            self.rhs = rhs
        else:
            self.rhs = rep_attr_rhs

        self.apply_rhs = self.rhs()
        self.time = 0.0
        self._history = []
        self.add_history()
        self._cds = None

    def step(self, time_step, noise=0.0):
        """
        Computes one time step with explicit Runge-Kutta method of 2nd order.
        By giving a noise parameter larger zero, this becomes an Ito-scheme
        for stochastic differential equations. As a user, normally you call
        either the :py:meth:`swarming.model.Model.evolve`,
        :py:meth:`swarming.model.Model.record_until` or
        :py:meth:`swarming.model.Model.record_for_time` to advance the system
        by a defined number of time steps.

        Parameters
        ----------
        time_step: float
            time step size to compute
        noise: float
            coefficient to apply an Ito diffusion to the system and turn it
            into a system of stochastic differential equations

        Returns
        -------
        :py:class:`swarming.model.Model`
            reference to class instance to allow for command chaining.
        """
        rhsx, rhsv = self.apply_rhs(self.X, self.V)
        X_half = self.X + .5 * time_step * rhsx
        V_half = self.V + .5 * time_step * rhsv

        rhsx, rhsv = self.apply_rhs(X_half, V_half)
        X = self.X + time_step * rhsx
        V = self.V + time_step * rhsv + np.sqrt(time_step) * noise * randn(self.X.shape[0], self.X.shape[1])

        # if update:
        self.X = X
        self.V = V
        self.time += time_step
        return self

    def evolve(self, time_step, n_steps, snapshot=False, **kwargs):
        """
        Get parameters for right hand side, get a right hand side for this
        parameter set and calculate a certain amount of time steps with it.

        Parameters
        ----------
        time_step: float
            time step size to compute
        n_steps: int
            number of time steps to calculate
        snapshot: bool
            wether or not to record the particle positions and velocities into
            the system history after calculating for ``n_steps`` time steps.
        kwargs: dict
            parameters for right hand side function

        Returns
        -------
        :py:class:`swarming.model.Model`
            reference to class instance to allow for command chaining.
        """
        if "noise" in kwargs.keys():
            noise = kwargs["noise"]
        else:
            noise = 0.0

        self.apply_rhs = self.rhs(**kwargs)
        for i in range(0, n_steps):
            self.step(time_step, noise=noise)

        if snapshot:
            self.add_history()

        return self

    def add_history(self):
        """
        Appends current simulation time, positions, velocities and mean
        positions and velocities into system history.
        """
        self._history.append(
            (self.time, self.X[:, 0], self.X[:, 1], self.V[:, 0], self.V[:, 1], self.mean_x, self.mean_v)
        )

    @property
    def history(self):
        return pd.DataFrame(self._history, columns=["time", "X1", "X2", "V1", "V2", "mean_x", "mean_v"])

    def record_for_time(self, max_time, time_step, n_steps, **kwargs):
        """
        Wrapper for :py:meth:`swarming.model.Model.evolve` to run batches of
        timesteps, until a certain maximum simulation time is reached,
        recording intermediate states into system history.

        Parameters
        ----------
        max_time: float
            maximum simulation time for which to compute. Adds to
            ``self.time``, so if ``self.time = 10`` and
            ``max_time = 20`` is given, ``self.time`` will be 30 in the end.
        time_step: float
            time step size to compute
        n_steps: int
            number of time steps to calculate for each recorded system state
        kwargs: dict
            parameters for right hand side function

        Returns
        -------
        :py:class:`swarming.model.Model`
            reference to class instance to allow for command chaining.
        """
        t = 0.0
        while t < max_time:
            self.evolve(time_step, n_steps, snapshot=True, **kwargs)
            t += time_step * n_steps

        return self

    def calc_chg_from_history(self, lookback=None):
        """
        For the recorded mean positions and velocities, calculate modulus of
        discrete change rates over time. Corresponds to modulus of discrete
        first derivatives of ``mean_x``, ``mean_v``

        Parameters
        ----------
        lookback: int
            Maximum amount of time points from history to take into account. If not
            given, use all.

        Returns
        -------
        chgx: numpy.array
            ``lookback`` sized array of mean position change rates
        chgv: numpy.array
            ``lookback`` sized array of mean velocity change rates
        time: numpy.array
            the timepoints for which change rates where calculated
        """
        if lookback is None or lookback > len(self._history):
            lookback = len(self._history)
        tim = self.history["time"].to_numpy()
        dt = tim[-lookback+1:] - tim[-lookback:-1]
        mx = np.stack(self.history["mean_x"], axis=0)
        chgx = ((mx[-lookback + 1:, :] - mx[-lookback:-1, :])**2).sum(axis=1) / dt
        mv = np.stack(self.history["mean_v"], axis=0)
        chgv = ((mv[-lookback + 1:, :] - mv[-lookback:-1, :])**2).sum(axis=1) / dt

        return chgx, chgv, tim[-lookback+1:]

    def record_until(
        self,
        max_steps,
        time_step,
        n_steps,
        lookback=None,
        tolx=1.e-2,
        tolv=1.e-2,
        **kwargs
    ):
        """
        Wrapper for :py:meth:`swarming.model.Model.evolve` to run batches of
        timesteps, until a maximum number of time steps has been reached,
        recording intermediate states into system history.

        Note
        ----
        It was intended to implement a stopping criterion involving mean
        velocities and position, but this is left as a future exercise :-)

        Parameters
        ----------
        max_steps: int
            maximum simulation steps to perform for one call of this function
        time_step: float
            time step size to compute
        n_steps: int
            number of time steps to calculate for each recorded system state
        lookback: int
            amount of previous time steps to take into account for not yet
            implemented stopping criterion
        tolx: float
            tolerance in position to take into account for not yet
            implemented stopping criterion
        tolv: float
            tolerance in velocity to take into account for not yet
            implemented stopping criterion
        kwargs: dict
            parameters for right hand side function

        Returns
        -------
        :py:class:`swarming.model.Model`
            reference to class instance to allow for command chaining.
        """
        
        indx = tolx + 1
        indv = tolv + 1
        count = 0
        while (count < max_steps):
            self.evolve(time_step, n_steps, snapshot=True, **kwargs)
            chgx, chgv, _ = self.calc_chg_from_history(lookback=lookback)
            indx = chgx.mean()
            indv = chgv.mean()
            count += 1

        return self

    @property
    def n_particles(self):
        return self.X.shape[0]

    @property
    def mean_x(self):
        return self.X.sum(axis=0) / self.n_particles

    @property
    def mean_v(self):
        return self.V.sum(axis=0) / self.n_particles

    @property
    def cds(self):
        if self._cds is None:
            self._cds = ColumnDataSource(data=self.cds_dict())
        return self._cds

    def update_cds(self):
        """
        Replaces the data dict of ColumnDataSource ``self.cds`` with current
        velocities and positions. If ``self.cds`` is attached as source to any
        bokeh figure, it will update, too.

        Returns
        -------
        :py:class:`swarming.model.Model`
            reference to class instance to allow for command chaining.
        """
        self._cds.data = self.cds_dict()
        return self

    def cds_dict(self):
        """
        Creates a data dictionary to insert into ColumnDataSource for current system state.
        See also :py:meth:`swarming.model.Model.cds_static`.

        Returns
        -------
        dict
            containing data for ColumnDataSource
        """
        return self.cds_static(self.X, self.V, vscale=self.VSCALE)

    @staticmethod
    def cds_static(X, V, vscale=1.0):
        """
        For given positions and velocities, create a dictionary that can be
        inserted into a bokeh ``ColumnDataSource``, providing data for
        plotting particle positions, as well as velocity vectors for a kind of
        quiver plot.

        Parameters
        ----------
        X: numpy.array
            ``n`` by 2 array containing in each "row" the current 2d positions of
            one of the ``n`` particles in the system
        V: numpy.array
            ``n`` by 2 array containing in each "row" the current 2d velocity
            vectors of one of the ``n`` particles in the system
        vscale: float
            factor by which velocity vectors will be stretched in
            visualization, for better visibility.

        Returns
        -------
        dict
            containing data for ColumnDataSource, with fields ``x1, x2``
            (positions), ``x1s, x2s`` (start and end coordinates of velocity
            vectors).
        """
        return dict(
            x1=X[:, 0],
            x2=X[:, 1],
            x1s=list(np.stack((X[:, 0], X[:, 0] + vscale * V[:, 0]), axis=1)),
            x2s=list(np.stack((X[:, 1], X[:, 1] + vscale * V[:, 1]), axis=1))
        )

    def plot(self, plot_width=500, plot_height=500, plot_mean=False):
        """
        Plot all particles with velocity vectors from current system state. Uses ``self.cds``.

        Parameters
        ----------
        plot_width: int
            plot width in pixels
        plot_height: int
            plot height in pixels
        plot_mean: bool
            mark the systems mean position and velocity vector or not.

        Returns
        -------
        bokeh.plotting.figure
            to be displayed with ``show`` or inserted into a layout.
        """

        f = figure(title="current state", plot_width=plot_width, plot_height=plot_height, match_aspect=True)
        f.circle(source=self.cds, x="x1", y="x2", size=13, fill_alpha=0.5)
        f.multi_line(source=self.cds, xs="x1s", ys="x2s", line_width=2)

        if plot_mean:
            mx = self.mean_x
            mv = self.mean_v
            f.circle_cross(x=mx[0], y=mx[1], size=15, fill_alpha=0.2, line_width=2, color="green")
            f.line(
                x=[mx[0], mx[0] + 10 * self.VSCALE * mv[0]],
                y=[mx[1], mx[1] + 10 * self.VSCALE * mv[1]],
                line_width=3,
                color="green"
            )
        return f

    def plot_density(self, plot_width=500, plot_height=400, palette="Viridis256", size=1):
        """
        Plot particle density from current system state.

        Parameters
        ----------
        plot_width: int
            plot width in pixels
        plot_height: int
            plot height in pixels
        palette: str or bokeh palette
            color palette or name of color palette to use
        size: float
            determines the size of hexagonal histogram cells in plot
            coordinates. 

        Returns
        -------
        bokeh.plotting.figure
            to be displayed with ``show`` or inserted into a layout.
        """
        f = figure(title="Density plot", plot_width=plot_width, plot_height=plot_height, match_aspect=True)
        _, counts = f.hexbin(x=self.X[:, 0], y=self.X[:, 1], size=size, palette=palette)
        colmapper = LinearColorMapper(low=counts["counts"].min(), high=counts["counts"].max(), palette=palette)
        colorbar = ColorBar(color_mapper=colmapper)
        f.background_fill_color = colmapper.palette[0]
        f.grid.visible = False

        f.add_layout(colorbar, "right")

        return f

    def plot_trajectory(self, plot_width=300, plot_height=300):
        """
        Plots the trajectory of the mean position and mean velocity of the
        whole system over time, recorded so far-

        Parameters
        ----------
        plot_width: int
            plot width in pixels
        plot_height: int
            plot height in pixels

        Returns
        -------
        bokeh.plotting.figure
            to be displayed with ``show`` or inserted into a layout.
        """
        x = np.stack(self.history["mean_x"], axis=0)
        v = np.stack(self.history["mean_v"], axis=0)

        f = figure(title="Mean trajectory", match_aspect=True, plot_width=plot_width, plot_height=plot_height)
        
        f.multi_line(
            xs=list(np.stack((x[:, 0], x[:, 0] + self.VSCALE * v[:, 0]), axis=1)),
            ys=list(np.stack((x[:, 1], x[:, 1] + self.VSCALE * v[:, 1]), axis=1)),
            color="blue"
        )
        f.line(x=x[:, 0], y=x[:, 1], color="green")
        return f


class InitialCondition(Model):
    """
    Subclass to prepare a system of interacting particles in a defined initial
    state or to reset it to one.
    Is based on :py:class:`swarming.model.Model`.

    Parameters
    ----------
    condition: str
        name of the initial condition to create. Must be a property name of
        this class
    n_particles: int
        number of particles to generate
    x_range: tuple
        x range where to distribute particles
    y_range: tuple
        y range where to distribute particles
    rhs: function
        function reference for right hand side for system, see
        :py:class:`swarming.model.Model`.

    Attributes
    ----------
    n: int
        given number of particles. Is effective when setting initial condition
    xr: tuple
        x range where to distribute particles
    yr: tuple
        y range where to distribute particles
    distx: float
        length of x range
    disty: float
        length of y range
    circular: tuple of np.array
        positions and velocities of a donut-shaped distribution of particles
    square: tuple of np.array
        positions and velocities of a square-shaped distribution of particles,
        all velocities aligned into one direction
    randomspeed: tuple of np.array
        like "square" with random velocity vectors
    nospeed: tuple of np.array
        like "square", with zero velocity vectors
    """
    def __init__(self, condition="circular", n_particles=100, x_range=(-40, 40), y_range=(-40, 40), rhs=None):
        self.n = n_particles
        self.xr = x_range
        self.yr = y_range
        super().__init__(*getattr(self, condition), rhs=rhs)

    @property
    def distx(self):
        return self.xr[1] - self.xr[0]
    
    @property
    def disty(self):
        return self.yr[1] - self.yr[0]

    @property
    def circular(self):
        halfdiam = 0.5 * self.distx
        r = 0.3*halfdiam + 0.3*halfdiam * rand(self.n)
        ang = 2*np.pi*rand(self.n)  # np.linspace(0, 2*np.pi, self.n)

        X = np.array([r * np.cos(ang), r*np.sin(ang)]).transpose()
        norm = np.sqrt(np.sum(X**2, 1))
        V = np.stack((-X[:, 1], X[:, 0]), axis=1) / np.stack((norm, norm), axis=1)
        return X, V

    @property
    def square(self):
        X = np.array([self.xr[0], self.yr[0]]) + rand(self.n, 2) * np.array([self.distx, self.disty])
        v = np.array([1, 1])
        V = np.array([v]*X.shape[0])
        return X, V

    @property
    def nospeed(self):
        X, V = self.square
        return X, np.zeros(V.shape)

    @property
    def randomspeed(self):
        X, V = self.square
        return X, 10. * (rand(*V.shape)-0.5)

    def set_initial(self, condition="circular"):
        """
        Prepare system in defined condition. Resets particle positions and
        velocities imminently.

        Returns
        -------
        :py:class:`swarming.model.Model`
            reference to class instance to allow for command chaining.
        """
        self.X, self.V = getattr(self, condition)
        return self

    def plot_initial(self, ncols=3, attrs=None, plot_width=300, plot_height=300):
        """
        Plot all registered initial conditions as an example. Does not change
        system state.

        Parameters
        ----------
        ncols: int
            number of columsn for grid plot of initial conditions
        attrs: list of str
            list of names of initial conditions to plot. All conditions are 
            lotted by default.
        plot_width: int
            plot width for each subfigure in pixels
        plot_height: int
            plot height for each subfigure in pixels

        Returns
        -------
        bokeh.layouts.grid
            grid of figures containing selected initial conditions.
        """
        if attrs is None:
            attrs = ["circular", "square", "nospeed", "randomspeed"]

        cds = [
            ColumnDataSource(
                data=self.cds_static(
                    *getattr(self, attr), vscale=self.VSCALE
                )
            )
            for attr in attrs
        ]

        figures = []
        for ini, source in zip(attrs, cds):
            f = figure(title=ini, plot_width=plot_width, plot_height=plot_height, match_aspect=True)
            f.circle(source=source, x="x1", y="x2")
            f.multi_line(source=source, xs="x1s", ys="x2s")
            figures.append(f)

        return grid(
            [figures[i: i+ncols] for i in range(0, len(figures), ncols)]
        )
