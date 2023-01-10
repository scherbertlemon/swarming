"""
Some helpers to conduct parameter studies for a given RHS and to evaluate them.
"""

import pandas as pd
from pyDOE import lhs
from bokeh.models import Div, LinearColorMapper, ColorBar
from bokeh.plotting import figure
from bokeh.layouts import column, grid
import numpy as np


DEFAULT_PARAMS = [
    ("Ca", 20, 10, 30),
    ("Cr", 50, 40, 60),
    ("la", 100, 40, 300),
    ("lr", 2, 1, 10),
    ("noise", 0.0, 0.0, 2.0)
]


class Parameters:
    """
    Creates lating hypercube sampling of given parameters in predefined ranges,
    using ``pyDOE``.

    Parameters
    ----------
    param_list: list of tuples
        List of 4-tuples, every tuple contains
        ``(parameter name, reference value, lower bound, upper bound)``,
        default value is the content of :py:const:`swarming.DEFAULT_PARAMS`.

    Attributes
    ----------
    params: pandas.DataFrame
        Dataframe containing the same information as given in parameter
        ``param_list``.
    n_params: int
        Number of parameters
    names: list of str
        List of parameter names, to be given to
        :py:func:`swarming.get_density_plots`.
    """
    def __init__(self, param_list=DEFAULT_PARAMS):
        self.params = pd.DataFrame(
            param_list,
            columns=["name", "value", "lower", "upper"]
        )

    @property
    def n_params(self):
        return len(self.params)

    @property
    def names(self):
        return self.params["name"].to_list()

    def sampling(self, n_samples=100):
        """
        Latin hypercube sampling for given number of parameters and number of
        samples.

        Parameters
        ----------
        n_samples: int
            number of samples to generate, default 100

        Returns
        -------
        numpy.array
            ``n_samples`` by ``n_params`` array with sampled values
            between 0, 1
        """
        return lhs(self.n_params, samples=n_samples)

    def sampling_dicts(self, n_samples=100):
        """
        Converts sampling to defined parameter range for each parameter.

        Parameters
        ----------
        n_samples: int
            number of samples to generate, default 100

        Returns
        -------
        list of dicts
            every dictionary has as keys the parameter names, values are the
            sampled values in the defined parameter ranges.
        """
        sdct = []
        smpl = self.sampling(n_samples=n_samples)
        for i in range(0, smpl.shape[0]):
            dct = {
                param["name"]: param["lower"] + (param["upper"] - param["lower"]) * value
                for param, value in zip(self.params.to_dict(orient="records"), smpl[i, :])
            }
            sdct.append(dct)
        return sdct

    def __repr__(self):
        return self.params.__repr__()  


def get_density_plots(df, ncols=2, param_names=None, plot_width=300,
                      plot_height=270, palette="Viridis256", size=1):
    """
    For a dataframe containing particle positions, calculates "density" by
    binning them in a 2-d historgram generated with
    :py:meth:`bokeh.plotting.figure.hexbin`. Furthermore display the
    parameters belonging to positions on top of the density plot.

    Parameters
    ----------
    df: pandas.DataFrame
        columns should contain the parameter names given in parameter
        ``param_names``, as well as ``X1`` and ``X2``, containing in each
        dataframe row x and y positions of all particles.
    ncols: int
        number of columns to order figures into
    param_names: list of str
        list of parameter names to extract from ``df`` columns to display.
    plot_width: int
        width of each figure in pixels.
    plot_height: int
        height of each figure in pixels.
    palette: str or palette from ``bokeh.palettes``
        color palette to indicate file counts in density plot
    size: float
        max distance of a point to center of hexagon, to be counted as
        contained in that hexagon (hexplot)

    Returns
    -------
    bokeh.layouts.grid
        grid of figures with density plots and parameter values, to be given
        to ``bokeh.io.show``
    """
    if param_names is None:
        param_names = Parameters().names
    figures = []
    for _, row in df.iterrows():
        params = [f"<li>{n} = {row[n]:.2f}</li>" for n in param_names]
        title = Div(text="<ul>{}</ul>".format("".join(params)))

        f = figure(title="Density plot", plot_width=plot_width, plot_height=plot_height, match_aspect=True)
        _, counts = f.hexbin(x=row["X1"], y=row["X2"], size=size, palette=palette)
        colmapper = LinearColorMapper(low=counts["counts"].min(), high=counts["counts"].max(), palette=palette)
        colorbar = ColorBar(color_mapper=colmapper)
        f.background_fill_color = colmapper.palette[0]
        f.grid.visible = False

        f.add_layout(colorbar, "right")
        figures.append(column(title, f))

    return grid(
        [figures[i:i+ncols] for i in range(0, len(figures), ncols)]
    )
