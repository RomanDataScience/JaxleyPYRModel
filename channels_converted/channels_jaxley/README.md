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
enable_cal4_diffusion(cell, axial_diffusion=0.22)
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
- `Cal4` is implemented as a Jaxley `Pump`, because it modifies the intracellular
  `CaCon_i` kinetic state. Call `enable_cal4_diffusion(cell, axial_diffusion=0.22)`
  or manually call `cell.diffuse("CaCon_i")` and set `axial_diffusion_CaCon_i`
  if you want longitudinal calcium diffusion. The original MOD file also uses
  NEURON radial diffusion and buffering constructs that are still reduced here.
- These files are importable and pass a small Jaxley insertion/integration smoke
  test, but they should still be validated against the original NEURON model
  before using them for scientific conclusions.
