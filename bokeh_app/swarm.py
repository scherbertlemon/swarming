from bokeh.io import curdoc
from bokeh.plotting import figure, ColumnDataSource
from time import sleep, time
from bokeh.models import Div, TextInput, Slider
from bokeh.layouts import column
import swarming
from tornado import gen

doc = curdoc()
TIME_STEP = 0.05
N_STEPS = 5
PARAMS = {
    "Ca": 20.0,
    "Cr": 50.0,
    "la": 100.0,
    "lr": 2.0
}

swarm = [swarming.InitialCondition(condition="square", n_particles=120)]

f = figure(title="Swarm", plot_width=500, plot_height=500, match_aspect=True)
f.multi_line(source=swarm[0].cds, xs="x1s", ys="x2s", line_width=4)

header = Div(text="{}ms".format(0.0))

@gen.coroutine
def update():
    start = time()
    swarm[0].cds.data = swarm[0].evolve(TIME_STEP, N_STEPS, **PARAMS).cds_dict()
    end = time()
    header.text = "{}ms".format((end-start)*1000.0)

doc.add_periodic_callback(update, 50)
doc.add_root(column(header, f))