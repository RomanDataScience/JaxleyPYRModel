# Combe2023 Jaxley Channels

This directory contains first-pass Jaxley translations for the active MOD
mechanisms inserted by `Combe2023/cell_setup_pc2b_CCh_driven.hoc`.

Each channel lives in its own Python module. Shared numerical helpers translated
from the MOD files, such as `MyExp`, GHK current, and exponential gate updates,
live in `common.py`.

Example:

```python
from channels_converted.channels_jaxley import Cal4, Kap, Nax, enable_cal4_diffusion

cell.apical.insert(Kap())
cell.axon.insert(Nax())
cell.insert(Cal4())
enable_cal4_diffusion(cell)
```

Shared parameters follow the current model convention where practical:

- `eNa`
- `eK`
- `eCa`
- `CaCon_i`
- `CaCon_e`
- `celsius`

Notes:

- `Nav16A` translates the MOD kinetic scheme with explicit Euler updates and an
  algebraic steady-state initialization.
- `Cal4` is a Jaxley `Pump` porting `cal4.mod` in full: four radial annuli
  scaled with the compartment diameter, the stationary buffer (rapid
  equilibrium), radial diffusion, the membrane pump and per-annulus ER
  exchange (SERCA, IP3 receptor with its inactivation gate, leak). Its
  parameters keep the MOD units (`KDs` in uM). `enable_cal4_diffusion(cell)`
  adds longitudinal diffusion of `CaCon_i` with the buffered coefficient
  (~0.005 um^2/ms). See `../CALCIUM_AND_KD.md`.
- Channels that read intracellular calcium (`Cal`, `Cat`, `Kca`, `MyKca`)
  declare `CaCon_i` in `channel_states` (`common.CALCIUM_STATE`). Jaxley only
  passes a channel the states it declares, so without this they silently read
  a constant. `Icand` does not read calcium, as in `icand.mod` (its `USEION ca`
  line is commented out).
- `Kdbm` is the slowly inactivating D-type K of Vitale et al. (2023)
  (`mod/kdbm.mod`, ModelDB 2014816), used in soma and dendrites of the Combe
  model when its `gkdbm_*` densities are set (0 by default).
- Every channel is validated against the compiled NEURON mechanism in
  `../validation` (`compare_channel.py` per channel with fixed calcium,
  `compare_calcium.py` for the coupled calcium system).
