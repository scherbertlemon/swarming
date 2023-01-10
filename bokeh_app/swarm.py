from bokeh.io import curdoc
from bokeh.plotting import figure, ColumnDataSource
from time import sleep, time
from bokeh.models import Slider, Dropdown, Toggle
from bokeh.layouts import column, row
import swarming
from tornado import gen

doc = curdoc()
TIME_STEP = 0.1
N_STEPS = 5
N_PARTICLES = 100
descriptor_params = {
    "Ca": [20.0, 5, 100, 1, "Attraction strength"],
    "Cr": [50.0, 5, 100, 1, "Repulsion strength"],
    "la": [100.0, 10, 300, .5, "Attraction range"],
    "lr": [2.0, 0.5, 10, 0.1, "Repulsion range"],
    "noise": [0.0, 0.0, 2, 0.1, "Noise"]
}
PARAMS = {key:value[0] for key, value in descriptor_params.items()}

# def get_float_from_text(txt):
#     try:
#         value = float(txt)
#     except ValueError:
#         print("Unnaccepable value {}".format(txt))
#         value = None
#     return value


swarm = [swarming.InitialCondition(condition="square", n_particles=N_PARTICLES)]


def slider_callback(name):
    def callback(attr, old, new):
        print(name, old, new)
        PARAMS[name] = new
    return callback


def ini_cond_callback(event):
    print("reset initial condition", event.item)
    swarm[0].set_initial(event.item)

for param, props in descriptor_params.items():
    props.append(Slider(start=props[1], end=props[2], value=props[0], step=props[3], title=props[4]))
    props[-1].on_change("value", slider_callback(param))

sim_toggle = Toggle(label="Run simulation on/off", button_type="success")
ini_cond = Dropdown(label="Initial condition", button_type="success", menu=[
    ("Donut", "circular"),
    ("Square, aligned velocity", "square"),
    ("Square, random velocity", "randomspeed"),
    ("Square, no velocity", "nospeed")
])
f = figure(title="Swarm", plot_width=600, plot_height=600, match_aspect=True)
f.circle(source=swarm[0].cds, x="x1", y="x2", size=13, fill_alpha=0.5)
f.multi_line(source=swarm[0].cds, xs="x1s", ys="x2s", line_width=2)

ini_cond.on_click(ini_cond_callback)

# def lr_input_callback(attr, old, new):
#     value = get_float_from_text(new)
#     if value:
#         PARAMS["lr"] = value
        
# lr_input.on_change("value", lr_input_callback)
#header = Div(text="{}ms".format(0.0))

@gen.coroutine
def update():
    # start = time()
    if sim_toggle.active:
        swarm[0].evolve(TIME_STEP, N_STEPS, **PARAMS).update_cds()
    # end = time()
    # header.text = "{}ms".format((end-start)*1000.0)

doc.add_periodic_callback(update, 50)
doc.add_root(column([props[-1] for _, props in descriptor_params.items()] + [row(ini_cond, sim_toggle), f]))