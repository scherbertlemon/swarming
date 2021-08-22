from datetime import time
import numpy as np
from numpy.random import rand, randn
from bokeh.plotting import ColumnDataSource, figure
from bokeh.layouts import grid
from bokeh.models import ColorBar, LinearColorMapper
import pandas as pd


def force_of_distance(r, Ca=20.0, Cr=50.0, la=100, lr=2, noise=0.0):
    return Ca/la*np.exp(-r/la)-Cr/lr*np.exp(-r/lr)


def harmonic_oscillator_rhs(const=0.01, damping=0.1, **kwargs):
    def apply_rhs(X, V):
        return V, - const* X - damping * V

    return apply_rhs


def rep_attr_rhs(alpha=0.07, beta=0.05, Ca=20.0, Cr=50.0, la=100, lr=2, noise=0.0, **kwargs):

    def apply_rhs(X, V):
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
        self._history.append(
            (self.time, self.X[:, 0], self.X[:, 1], self.V[:, 0], self.V[:, 1], self.mean_x, self.mean_v)
        )
        
    
    @property
    def history(self):
        return pd.DataFrame(self._history, columns=["time", "X1", "X2", "V1", "V2", "mean_x", "mean_v"])

    def record_for_time(self, max_time, time_step, n_steps, **kwargs):
        t = 0.0
        while t < max_time:
            self.evolve(time_step, n_steps, snapshot=True, **kwargs)
            t += time_step * n_steps

        return self

    def calc_chg_from_history(self, lookback=None):
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
        self._cds.data = self.cds_dict()
        return self

    def cds_dict(self):
        return self.cds_static(self.X, self.V, vscale=self.VSCALE)

    @staticmethod
    def cds_static(X, V, vscale=1.0):
        return dict(
            x1=X[:, 0],
            x2=X[:, 1],
            x1s=list(np.stack((X[:, 0], X[:, 0] + vscale * V[:, 0]), axis=1)),
            x2s=list(np.stack((X[:, 1], X[:, 1] + vscale * V[:, 1]), axis=1))
        )

    def plot(self, plot_width=500, plot_height=500, plot_mean=False):
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

    def plot_density(self, ncols=2, plot_width=500, plot_height=400, palette="Viridis256", size=1):
        
        f = figure(title="Density plot", plot_width=plot_width, plot_height=plot_height, match_aspect=True)
        _, counts = f.hexbin(x=self.X[:, 0], y=self.X[:, 1], size=size, palette=palette)
        colmapper = LinearColorMapper(low=counts["counts"].min(), high=counts["counts"].max(), palette=palette)
        colorbar = ColorBar(color_mapper=colmapper)
        f.background_fill_color = colmapper.palette[0]
        f.grid.visible = False

        f.add_layout(colorbar, "right")

        return f

    def plot_trajectory(self, plot_width=300, plot_height=300):

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
        self.X, self.V = getattr(self, condition)
        return self

    def plot_initial(self, ncols=3, attrs=None, plot_width=300, plot_height=300):
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