"""Jaxley port of ``cal4.mod`` (Fink et al. 2000 calcium dynamics, as used by Combe 2023).

``cal4.mod`` models intracellular calcium in four radial annuli per segment:

* calcium entry from the membrane calcium current into the outermost annulus,
* a membrane pump on the outermost annulus (active only above ``cath``),
* radial diffusion between annuli (``DCa``),
* a stationary buffer in every annulus (``TBufs``, ``KDs``; BAPTA is 0),
* per annulus ER exchange: SERCA uptake, IP3-receptor release with its
  inactivation gate ``ho``, and a leak ``L`` balancing both at rest,
* longitudinal diffusion of free calcium along the neurite.

``cai`` is the free calcium of the outermost annulus. This port keeps all of the
above with two numerical reductions:

* **Rapid buffer equilibrium.** The stationary buffer binds at
  ``kfs * TBufs + kfs * KDs = 460 /ms`` (time constant ~2 us), far faster than
  the simulation step, so it is treated as always in equilibrium: every flux
  into an annulus is divided by ``1 + TBufs * KDs / (KDs + ca)^2`` (~46 at rest).
* **Longitudinal diffusion** acts only on the outermost annulus (``CaCon_i``),
  through Jaxley's ``cell.diffuse("CaCon_i")``; see ``enable_cal4_diffusion``,
  which uses the buffered coefficient ``DCa / (1 + TBufs * KDs / (KDs + cai)^2)``.

The electrogenic pump current ``ica_pmp`` that ``cal4.mod`` writes into ``ica``
is not added to the membrane current (a Jaxley pump cannot carry one). It is
``0.154 * (cai - cath)`` mA/cm2 with cai in mM, zero while ``cai < cath``
(0.2 uM): about 1.5e-4 mA/cm2 at cai = 1 uM and 1.5e-3 mA/cm2 at 10 uM.

**Integration.** ``update_states`` advances the four annuli together with one
implicit Euler step (radial diffusion, calcium entry and the membrane pump
implicit; ER exchange and the buffering factor evaluated at the start of the
step), mirroring the fully coupled sparse solve of the MOD file, and writes
``CaCon_i`` (outer annulus) and ``cal4_ca1``..``cal4_ca3``. The IP3-receptor
gates ``cal4_ho0``..``cal4_ho3`` use their exact update. ``compute_current``
then returns zero, so Jaxley's own ``CaCon_i`` step only applies longitudinal
diffusion. The calcium current used is the one from the previous step, as in
NEURON.

Validated against the compiled NEURON mechanism with
``channels_converted/validation/compare_calcium.py``.
"""

import math

import jax.numpy as jnp
from jaxley.pumps import Pump

from .common import channel_prefix

N_ANNULI = 4
# cal4.mod uses FARADAY = (faraday) (10000 coulomb).
FARADAY_MOD = 96485.33212 / 1.0e4


def _annulus_factors(n: int = N_ANNULI):
    """``factors()`` of cal4.mod: volume and diffusion factors for diam = 1."""
    r = 0.5
    dr2 = r / (n - 1) / 2.0
    vrat = [0.0] * n
    frat = [0.0] * n
    frat[0] = 2.0 * r
    for i in range(n - 1):
        vrat[i] += math.pi * (r - dr2 / 2.0) * 2.0 * dr2
        r -= dr2
        frat[i + 1] = 2.0 * math.pi * r / (2.0 * dr2)
        r -= dr2
        vrat[i + 1] = math.pi * (r + dr2 / 2.0) * 2.0 * dr2
    return tuple(vrat), tuple(frat)


VRAT, FRAT = _annulus_factors()


class Cal4(Pump):
    """Four-annulus buffered calcium with ER exchange (``cal4.mod``)."""

    def __init__(self, name=None):
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_ip3i": 10e-3,
            f"{prefix}_cai0": 50e-6,
            f"{prefix}_cath": 0.2e-3,
            f"{prefix}_gamma": 8.0,
            f"{prefix}_jmax": 3.5e-3,
            f"{prefix}_caer": 0.400,
            f"{prefix}_Kip3": 0.8e-3,
            f"{prefix}_Kact": 0.7e-3,
            f"{prefix}_kon": 2.7,
            f"{prefix}_Kinh": 0.6e-3,
            f"{prefix}_sites": 3.0,
            f"{prefix}_alpha": 1.0,
            f"{prefix}_beta": 1.0,
            f"{prefix}_vmax": 1e-4,
            f"{prefix}_Kp": 0.27e-3,
            f"{prefix}_DCa": 0.22,
            f"{prefix}_TBufs": 0.45,
            # Units as in cal4.mod: KDs in uM (converted to mM below).
            f"{prefix}_KDs": 10.0,
        }
        self.channel_states = {
            # cal4.mod initializes the calcium current to zero.
            "i_Ca": 0.0,
            "CaCon_i": 50e-6,
            **{f"{prefix}_ca{i}": 50e-6 for i in range(1, N_ANNULI)},
            **{f"{prefix}_ho{i}": 0.0 for i in range(N_ANNULI)},
        }
        self.ion_name = "CaCon_i"
        self.current_name = "i_Cal4"

    # ------------------------------------------------------------------ helpers
    def _buffer_factor(self, ca, params):
        """1 + d[CaBuf]/d[Ca] for the stationary buffer in equilibrium."""
        prefix = channel_prefix(self)
        kd = 1e-3 * params[f"{prefix}_KDs"]  # uM -> mM
        return 1.0 + params[f"{prefix}_TBufs"] * kd / (kd + ca) ** 2

    def _ip3_open(self, ca, ho, params):
        prefix = channel_prefix(self)
        ip3 = params[f"{prefix}_ip3i"] / (params[f"{prefix}_ip3i"] + params[f"{prefix}_Kip3"])
        act = ca / (ca + params[f"{prefix}_Kact"])
        return (ip3 * act * ho) ** params[f"{prefix}_sites"]

    def _leak_rate(self, params):
        """L of cal4.mod: balances SERCA and IP3R at ca = cai0, ho at rest."""
        prefix = channel_prefix(self)
        ca = params[f"{prefix}_cai0"]
        ho = params[f"{prefix}_Kinh"] / (ca + params[f"{prefix}_Kinh"])
        jx = -params[f"{prefix}_vmax"] * ca**2 / (ca**2 + params[f"{prefix}_Kp"] ** 2)
        jx = jx + params[f"{prefix}_jmax"] * (1.0 - ca / params[f"{prefix}_caer"]) * self._ip3_open(
            ca, ho, params
        )
        return -jx / (1.0 - ca / params[f"{prefix}_caer"])

    def _er_rate(self, ca, ho, params):
        """SERCA + IP3R + leak for one annulus, in mM/ms of free-plus-bound calcium."""
        prefix = channel_prefix(self)
        caer = params[f"{prefix}_caer"]
        serca = params[f"{prefix}_beta"] * params[f"{prefix}_vmax"] * ca**2 / (
            ca**2 + params[f"{prefix}_Kp"] ** 2
        )
        release = (
            params[f"{prefix}_alpha"]
            * params[f"{prefix}_jmax"]
            * (1.0 - ca / caer)
            * self._ip3_open(ca, ho, params)
        )
        leak = params[f"{prefix}_beta"] * self._leak_rate(params) * (1.0 - ca / caer)
        return release + leak - serca

    @staticmethod
    def _diam(params):
        return 2.0 * params["radius"]

    # ------------------------------------------------------------------ Jaxley API
    def update_states(self, states, dt, v, params):
        """One coupled implicit step of the four annuli, plus the IP3R gates."""
        prefix = channel_prefix(self)
        kon = params[f"{prefix}_kon"]
        kinh = params[f"{prefix}_Kinh"]
        ca = [states["CaCon_i"]] + [states[f"{prefix}_ca{i}"] for i in range(1, N_ANNULI)]
        ca = [jnp.maximum(c, 1e-12) for c in ca]
        ho = [states[f"{prefix}_ho{i}"] for i in range(N_ANNULI)]

        diam = self._diam(params)
        dsq = diam * diam
        k = [params[f"{prefix}_DCa"] * FRAT[i + 1] for i in range(N_ANNULI - 1)]
        # ~ ca[0] << (-(ica - ica_pmp_last) * PI * diam / (2 * FARADAY)); i_Ca is
        # the summed calcium-channel current (the pump has its own current name).
        influx = -states["i_Ca"] * math.pi * diam / (2.0 * FARADAY_MOD)
        # ~ ca[0] <-> sump, both rates 0.001 * parea * gamma * u(ca0 > cath).
        pump = jnp.where(ca[0] > params[f"{prefix}_cath"],
                         1e-3 * math.pi * diam * params[f"{prefix}_gamma"], 0.0)

        rows, rhs = [], []
        for i in range(N_ANNULI):
            volume = dsq * VRAT[i]
            cap = volume * self._buffer_factor(ca[i], params) / dt
            k_out = k[i - 1] if i > 0 else 0.0
            k_in = k[i] if i < N_ANNULI - 1 else 0.0
            diag = cap + k_out + k_in + (pump if i == 0 else 0.0)
            row = [jnp.zeros_like(diam) for _ in range(N_ANNULI)]
            row[i] = diag + jnp.zeros_like(diam)
            if i > 0:
                row[i - 1] = -k_out + jnp.zeros_like(diam)
            if i < N_ANNULI - 1:
                row[i + 1] = -k_in + jnp.zeros_like(diam)
            rows.append(jnp.stack(row, axis=-1))
            source = cap * ca[i] + volume * self._er_rate(ca[i], ho[i], params)
            if i == 0:
                source = source + influx + pump * params[f"{prefix}_cath"]
            rhs.append(source)
        solved = jnp.linalg.solve(jnp.stack(rows, axis=-2),
                                  jnp.stack(rhs, axis=-1)[..., None])[..., 0]

        updated = {"i_Ca": states["i_Ca"],
                   "CaCon_i": jnp.maximum(solved[..., 0], 1e-12)}
        for i in range(1, N_ANNULI):
            updated[f"{prefix}_ca{i}"] = jnp.maximum(solved[..., i], 1e-12)
        # hc <-> ho (kon*Kinh, kon*ca): exact update at the start-of-step calcium.
        for i in range(N_ANNULI):
            rate = kon * (kinh + ca[i])
            ho_inf = kinh / (kinh + ca[i])
            updated[f"{prefix}_ho{i}"] = ho_inf + (ho[i] - ho_inf) * jnp.exp(-dt * rate)
        return updated

    def compute_current(self, states, modified_state, params):
        """No further change: update_states already advanced CaCon_i.

        Jaxley's implicit CaCon_i step then applies only longitudinal
        diffusion (when enabled).
        """
        return jnp.zeros_like(modified_state)

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        ca = params[f"{prefix}_cai0"]
        ho = params[f"{prefix}_Kinh"] / (ca + params[f"{prefix}_Kinh"])
        return {
            "CaCon_i": ca,
            **{f"{prefix}_ca{i}": ca for i in range(1, N_ANNULI)},
            **{f"{prefix}_ho{i}": ho for i in range(N_ANNULI)},
        }


def buffered_diffusion_coefficient(DCa=0.22, TBufs=0.45, KDs=10.0, cai=50e-6):
    """Longitudinal coefficient of free calcium under the stationary buffer (um^2/ms).

    Units as in cal4.mod: DCa in um^2/ms, TBufs and cai in mM, KDs in uM.
    """
    kd = 1e-3 * KDs
    return DCa / (1.0 + TBufs * kd / (kd + cai) ** 2)


def enable_cal4_diffusion(cell, axial_diffusion=None):
    """Enable longitudinal diffusion of ``CaCon_i`` (the outer annulus of ``Cal4``).

    ``cal4.mod`` diffuses free calcium longitudinally with ``DCa = 0.22 um^2/ms``
    while the stationary buffer binds most of it. With the buffer in rapid
    equilibrium the free calcium diffuses with ``DCa / (1 + TBufs * KDs / (KDs +
    cai)^2)`` (~0.005 um^2/ms at rest), which is the default here. Pass
    ``axial_diffusion`` to override.
    """
    if axial_diffusion is None:
        axial_diffusion = buffered_diffusion_coefficient()
    if axial_diffusion <= 0.0:
        raise ValueError("axial_diffusion must be strictly positive.")

    diffusion_states = getattr(cell, "diffusion_states", None)
    if diffusion_states is None and hasattr(cell, "base"):
        diffusion_states = getattr(cell.base, "diffusion_states", [])

    if "CaCon_i" not in (diffusion_states or []):
        cell.diffuse("CaCon_i")
    cell.set("axial_diffusion_CaCon_i", axial_diffusion)
    return cell
