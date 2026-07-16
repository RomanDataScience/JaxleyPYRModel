from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/jaxley_mpl")

import matplotlib.pyplot as plt

import jaxley as jx

# To suppress Pandas performance warnings:
import pandas as pd
import warnings
warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)

from model import L5PC, add_test_stimuli, add_segmented_stimuli

delta_t = 0.025

cell = L5PC()

# cell, time_vec = add_test_stimuli(cell)
cell, time_vec, stimulus = add_segmented_stimuli(
    cell,
    cell_name="m20240527cd",
    trace_name="v75ctrl",
    segment_name="hyperpolarizing_step",
    experimental_dt=0.05,
    delta_t=delta_t,
)


x_o = jx.integrate(cell, delta_t=delta_t)[0]  # [0] gets rid of the batch-dimension.

fig, ax = plt.subplots(2, 1, figsize=(5.0, 2.0))
_ = ax[0].plot(time_vec, x_o, c="k")
_ = ax[0].set_ylim([-90, 60])
_ = ax[0].set_xlabel("Time (ms)")
_ = ax[0].set_ylabel("Voltage (mV)")

_ = ax[1].plot(time_vec[:len(stimulus)], stimulus, c="b")
_ = ax[1].set_xlabel("Time (ms)")
_ = ax[1].set_ylabel("Current (nA)")
# plt.savefig("trace.png")
plt.show()
