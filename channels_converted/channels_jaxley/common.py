import jax.numpy as jnp
from jaxley.channels import Channel


FARADAY = 96485.33212
FARADAY_KC = 96.48533212
R = 8.314462618
EPS = 1e-12


def safe_exp(x):
    return jnp.where(x < -50.0, 0.0, jnp.exp(jnp.minimum(x, 50.0)))


def sigmoid_arg(arg):
    return 1.0 / (1.0 + safe_exp(arg))


def inv_sigmoid_arg(arg):
    return 1.0 / (1.0 + safe_exp(-arg))


def positive(x, floor=EPS):
    return jnp.maximum(x, floor)


def gate_update(x, dt, x_inf, tau):
    tau = positive(tau)
    exp_term = safe_exp(-dt / tau)
    return x * exp_term + x_inf * (1.0 - exp_term)


def vtrap(x, y):
    arg = x / y
    denom = safe_exp(arg) - 1.0
    raw = x / jnp.where(jnp.abs(denom) < EPS, EPS, denom)
    return jnp.where(jnp.abs(arg) < 1e-6, y, raw)


def trap0(v, th, a, q):
    arg = -(v - th) / q
    denom = 1.0 - safe_exp(arg)
    raw = a * (v - th) / jnp.where(jnp.abs(denom) < EPS, EPS, denom)
    return jnp.where(jnp.abs(v - th) > 1e-6, raw, a * q)


def efun(z):
    denom = safe_exp(z) - 1.0
    raw = z / jnp.where(jnp.abs(denom) < EPS, EPS, denom)
    return jnp.where(jnp.abs(z) < 1e-4, 1.0 - z / 2.0, raw)


def ghk(v, ci, co, celsius):
    ktf = (25.0 / 293.15) * (celsius + 273.15)
    f = ktf / 2.0
    nu = v / f
    return -f * (1.0 - (ci / co) * safe_exp(nu)) * efun(nu)


def q10_factor(q10, celsius, reference):
    return q10 ** ((celsius - reference) / 10.0)


def state_or_param(states, params, key, default):
    if key in states:
        return states[key]
    if key in params:
        return params[key]
    return default


def channel_prefix(channel):
    return channel._name

