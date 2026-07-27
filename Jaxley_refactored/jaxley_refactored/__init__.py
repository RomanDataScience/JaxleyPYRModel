"""Modular Combe2023 construction, simulation, and fitting.

The top-level package deliberately does not import JAX. Command-line programs
first apply the requested device and precision policy, then import numerical
modules. This keeps library imports safe on CPU, GPU, and SLURM workers.
"""

__version__ = "0.1.0"

