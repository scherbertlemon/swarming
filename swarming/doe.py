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

    def __init__(self, param_list=DEFAULT_PARAMS):
        self.params = pd.DataFrame(param_list, columns=["name", "value", "lower", "upper"])
        
    @property
    def n_params(self):
        return len(self.params)

    @property
    def names(self):
        return self.params["name"].to_list()

    def sampling(self, n_samples=100):
        return lhs(self.n_params, samples=n_samples)

    def sampling_dicts(self, n_samples=100):
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


def get_density_plots(df, ncols=2, param_names=None, plot_width=300, plot_height=270, palette="Viridis256", size=1):
    
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
