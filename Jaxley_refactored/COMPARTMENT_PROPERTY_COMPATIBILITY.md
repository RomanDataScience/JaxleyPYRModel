# HOC-compatible compartment-property updates

Status: implemented in `JaxleyModel/model/model_Combe.py` on 2026-07-27.

## Resolved issue

Changing compartment properties is necessary when a fitted model parameter
changes. The defect was that `set_fitted_parameters()` recomputed every passive
and active profile, including unrelated profiles, before the first optimizer
step. On the default 144-branch, 974-compartment HOC model, even an empty update
changed capacitance, axial resistivity, leak, and multiple channel arrays.

The corrected invariant is:

```text
no selected parameters       -> no param_state writes
reference parameter values   -> bitwise-identical HOC baseline
selected parameter values    -> only declared dependent targets change
```

## Why HOC endpoint distance is required

The legacy setup starts with one segment per section. It assigns most fitted
passive and active profiles, then changes `nseg` several times. The last write
of each original `for (x)` loop is at the section endpoint, so these values are
sectionwise constants evaluated at:

```python
h.distance(1.0, sec=section)
```

NEURON reuses/copies old segment values when `nseg` changes; it does not
numerically interpolate nonconstant RANGE values. Its documentation recommends
re-executing the assignment expression when such profiles must follow a new
grid:

https://nrn.readthedocs.io/en/latest/progref/modelspec/programmatic/topology.html#nseg

Reconstructing the assignment coordinate from Jaxley's ordinary path distance
is not exact. All 51 basal sections differ by one soma length because their HOC
roots attach at `soma(0)`, while the reduced parent-only Jaxley tree routes path
distance through the soma branch end. The importer now stores
`hoc_assignment_distance_um` directly from live NEURON for every final
compartment.

## Implemented update modes

### `exact_hoc_frozen_grid`

This is the default for `morphology_source="hoc"` with the reference Combe
parameters. For every dirty target:

```text
target(theta)
  = immutable_hoc_reference
    + endpoint_rule(theta)
    - endpoint_rule(theta_reference)
```

The complete nonlinear rule is evaluated from the full parameter vector.
Consequently, products such as `gna * scale_Na_conduct`,
`gna * AXNa`, `gnadend * scale_Na_conduct`, and
`gkv2 * gkv2scale` retain their cross terms. Zero-reference conductances are
also supported without division.

This is exact for the legacy assignment semantics on the frozen 974-compartment
grid and remains compatible with `jax.jit`, automatic differentiation, GPU
execution, and a future `vmap` trace backbone.

### `rule_based_final_centers`

SWC and custom-rule builds evaluate the portable distribution rules at each
final Jaxley compartment center. They do not use HOC endpoint coordinates.

### Future `hoc_rebuild`

Seven passive knobs can change the legacy `d_lambda` discretization:

- `CmSoma`
- `SpineFactorBasal`
- `SpineFactorTuft`
- `RaSoma`
- `RaTuft`
- `DistHalfRa`
- `SlopeRa`

A true fresh-HOC rebuild may therefore change `ncomp` discontinuously. Dynamic
rediscretization cannot live inside a fixed-shape JIT/gradient optimization
step. A future CPU-only `hoc_rebuild` mode should be used as an outer-loop
validation oracle when that structural behavior is required.

## Selective dependency behavior

The current 40 fitted keys now have explicit target dependencies. Examples:

- `soma_hbar` writes only soma, apical, and basal `h_gbar`.
- `RmSoma` writes only regional `Leak_gLeak`.
- `SpineFactorBasal` writes only basal capacitance and leak.
- `gna` writes only somatic `na16a_gbar` and axonal `nax_gbar`.
- `gkv2scale` writes only apical and basal `Kv2like_gbar`.
- `Epas` writes only `Leak_eLeak`.

The fitter no longer overwrites `na16a_dist` or `na16a_C1O1v2`; neither is
controlled by the current fit-key catalog.

## Validation

Run:

```bash
NEURON_MODULE_OPTIONS=-nogui \
MPLBACKEND=Agg \
MPLCONFIGDIR=/tmp/jaxley-mpl \
/Users/romanbaravalle/miniconda3/envs/Jaxley/bin/python -m pytest -q \
  JaxleyModel/model/tests/test_model_combe_parameter_updates.py
```

Current result:

```text
50 passed
```

The suite covers:

- Empty-update identity.
- Bitwise identity for all 40 reference values.
- Exhaustive target isolation and a real property change for all 40 knobs.
- Exported HOC endpoint semantics and sectionwise profiles.
- Simultaneous nonlinear passive updates.
- Coupled conductance products.
- Activation of zero-default conductances.
- JIT compilation and finite gradients.
- SWC final-center rule behavior.
- Invalid request validation.

An additional short end-to-end `jax.jit` integration check completed with 11
finite voltage samples and a finite gradient through `soma_hbar`.
