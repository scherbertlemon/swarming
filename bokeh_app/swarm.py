from bokeh.io import curdoc
from bokeh.plotting import figure, ColumnDataSource
from time import sleep, time
from bokeh.models import Div, TextInput, Slider
from bokeh.layouts import column
import swarming
from tornado import gen

doc = curdoc()
args = doc.session_context.request.arguments

ts = .2


ini = swarming.InitialCondition(n_particles=n_particles)
swarm = swarming.Model(*ini.random_speed, rhs=swarming.rep_attr_rhs)

cds = ColumnDataSource(data=swarm.cds_dict())
f = figure(title="Swarm", plot_width=500, plot_height=500, match_aspect=True)
f.multi_line(source=cds, xs="x1s", ys="x2s", line_width=4)

header = Div(text="{}ms".format(0.0))

@gen.coroutine
def update():
    start = time()
    cds.data = swarm.evolve(ts).cds_dict()
    end = time()
    header.text = "{}ms".format((end-start)*1000.0)


# one test evolve
start = time()
swarm.evolve(ts).cds_dict()
duration = (time()-start) * 1000
print(f"comp time one step: {duration}ms")

doc.add_periodic_callback(update, duration + 10)
doc.add_root(column(header, f))