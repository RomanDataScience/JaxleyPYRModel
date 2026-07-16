from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxley as jx
from jaxley.channels import Leak
from jaxley_mech.channels.l5pc import *
from jaxley.morphology import distance_direct

# To suppress Pandas performance warnings:
import pandas as pd
import warnings
warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)

from model import L5PC

cell = L5PC()

x_o = jx.integrate(cell)[0]  # [0] gets rid of the batch-dimension.

fig, ax = plt.subplots(1, 1, figsize=(5.0, 2.0))
_ = ax.plot(time_vec, x_o, c="k")
_ = ax.set_ylim([-90, 60])
_ = ax.set_xlabel("Time (ms)")
_ = ax.set_ylabel("Voltage (mV)")
plt.show()

