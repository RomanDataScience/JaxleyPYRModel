# HOC to Jaxley Import Notes

These notes summarize the issues found while porting
`Combe2023/cell_setup_pc2b_CCh_driven.hoc` into Jaxley. The main lesson is
that matching channel equations is necessary but not sufficient: morphology
topology, discretization, passive load, ion reversals, and final per-segment
channel placement also have to match.

## Reproduction Target

When the goal is to reproduce an existing HOC model, treat NEURON's final
section/segment table as the reference, not an intermediate SWC conversion or a
handwritten copy of the formulas.

For the current Combe port, the exact-reference path is:

- Build the morphology from the loaded HOC section tree.
- Use one Jaxley branch per HOC section.
- Use the same `d_lambda` in NEURON and Jaxley.
- Copy final NEURON per-segment geometry, passive values, channel parameters,
  ion reversals, and `celsius` into Jaxley.
- Validate with passive-only tests before running the full active model.

The implementation lives in `JaxleyModel/model/model_Combe.py`:

- `build_hoc_section_cell()`
- `apply_hoc_passive_profile()`
- `apply_hoc_channel_profile()`

## Issues Found

### 1. SWC topology is not equivalent to HOC topology

A standard SWC cannot always represent NEURON's zero-length logical section
connections. In the Combe HOC model, some child sections are connected
logically to a parent section even when the child `pt3d` coordinates do not
start at the parent's endpoint.

If this is forced into SWC, the converter can create artificial cable between
the parent and child coordinates. That changes surface area, capacitance, axial
load, and therefore the voltage response.

Fix used here:

- Read the HOC model in NEURON.
- Traverse `soma[0]` and its `sec.children()` topology.
- Create one Jaxley `Branch` per HOC section.
- Use each HOC section's own `pt3d` geometry without adding artificial parent
  connector cable.

Implication:

- If the goal is to reproduce the HOC model, the HOC section tree is the
  correct morphology source.
- `CELL.SWC` can still be useful for Jaxley-only experiments, but it should not
  be assumed to be an exact representation of the HOC morphology.

### 2. Passive RANGE variables can be interpolated across `nseg` changes

The HOC file assigns passive properties such as `cm`, `Ra`, `g_pas`, and
`e_pas` while NEURON has a particular segment layout. Later, `nseg` can be
changed again. NEURON preserves/interpolates RANGE variables into the final
segment layout.

That means the final per-segment passive values are not always identical to
evaluating the HOC formula directly at the final Jaxley compartment centers.

Fix used here:

- Copy final NEURON values into Jaxley:
  - `seg.cm`
  - `sec.Ra`
  - `seg.pas.g`
  - `seg.pas.e`
- Validate group totals:
  - total surface area
  - `cm * area`
  - `g_pas * area`

Useful diagnostic order:

1. Leak-only, no active channels.
2. Passive morphology with step current.
3. Full active model.

### 3. Active channel placement has the same interpolation issue

Several active parameters are assigned in the HOC file using distance-dependent
or section-dependent formulas, then the final segment table can differ from the
assignment-time table.

Examples:

- `h.gbar`
- `kap.gkabar`
- `kad.gkabar`
- `Kv2like.gbar`
- `na16a.dist`
- `na16a.C1O1v2`
- `kir.gbar`
- `cal4.alpha`

Initially, reimplementing the formulas in Jaxley got close but did not exactly
match the HOC model, especially at higher current amplitudes. Copying the final
NEURON per-segment active parameters removed most of that mismatch.

Fix used here:

- Build a HOC-to-Jaxley parameter map from the Jaxley channel parameter names.
- Store final NEURON values as `hoc_*` columns.
- After inserting Jaxley channels, copy `hoc_*` values into the actual Jaxley
  parameter columns where both are finite.

### 4. Ion reversals are part of the model state

Do not assume a single common value for every ion reversal just because most
channels share names like `eNa`, `eK`, and `eCa`.

In this model, copying the final NEURON ion values mattered. For example,
`eK` was not uniform across all final compartments after the HOC setup.

Fix used here:

- Copy final NEURON segment values:
  - `ena` to `eNa`
  - `ek` to `eK`
  - `eca` to `eCa`
- Copy `h.celsius` into Jaxley `celsius`.

### 5. Distance conventions must be explicit

The HOC model uses `soma distance()` and then calls `distance(x)` inside
section loops. Jaxley distance calculations must use the same root and path
convention.

Fix used here:

- Use path distance from soma, not direct Euclidean distance.
- Set the Jaxley root at `soma.branch(0).comp(0)`.
- Add the soma compartment half-length offset so compartment-center distances
  match the HOC convention more closely.

For a new morphology, re-check distance values before assigning any
distance-dependent conductance.

### 6. Section groups are model semantics, not just morphology labels

The HOC model applies different channel rules to soma, axon, basal dendrites,
apical dendrites, and sometimes apical trunk/non-trunk sections.

If a new morphology is imported from SWC, the SWC type labels may not be enough
to reproduce the HOC section grouping. You need a deliberate group assignment
step.

Minimum groups needed by the current Jaxley Combe port:

- `soma`
- `axon`
- `basal`
- `apical`

For future morphologies, decide how to classify apical trunk versus non-trunk
before reusing any trunk-specific HOC rules.

### 7. `d_lambda`, `nseg`, and `dt` must be synchronized

When comparing NEURON and Jaxley:

- Use the same `d_lambda` in both models.
- Recompute NEURON `nseg` after changing `d_lambda`.
- Rebuild the Jaxley morphology/compartment count after changing `d_lambda`.
- Use the same `dt` for both integrations.
- Rebuild the stimulus if `dt` changes.

Important Jaxley stimulus caveat:

The n-th entry of a Jaxley stimulus vector is applied at the n-th simulation
step. If `jx.integrate(..., delta_t=...)` changes, the stimulus must be rebuilt
or resampled for that same `delta_t`.

### 8. Channel dynamics validation is separate from model validation

A channel can match the MOD file in isolation and still produce a different
cell response if it is placed on the wrong compartments or uses the wrong
passive/ion environment.

Recommended validation ladder:

1. Isolated channel validation against NEURON under voltage clamp.
2. Single-compartment insertion smoke test in Jaxley.
3. Full morphology passive-only comparison.
4. Full morphology with one channel family enabled.
5. Full active model at subthreshold current.
6. Full active model at spike-generating current.

Avoid validating channels only at their default conductance. Some mechanisms
have `gbar = 0` by default in isolated contexts, so a current trace can look
correct while testing no current at all.

### 9. `cal4` is not a direct one-line channel translation

The original `cal4.mod` uses NEURON kinetic/radial calcium machinery. The
Jaxley version is implemented as a `Pump` that owns the `CaCon_i` update, with
optional longitudinal diffusion through:

```python
enable_cal4_diffusion(cell, axial_diffusion=0.22)
```

This is the right Jaxley shape for concentration dynamics, but it should be
validated separately from ordinary voltage-gated channels.

### 10. Numerical details can hide as biological differences

The current port requires:

- JAX x64 enabled.
- CPU execution for reproducible comparison runs.
- A compatibility patch for the local Jaxley/JAX `save_exp` call.

After morphology, passive load, active placement, ion reversals, and x64 were
aligned, the remaining high-current difference was mostly a one-step spike
upstroke timing offset, not a large conductance-placement error.

## Future Morphology Checklist

Use this checklist when applying the same channel dynamics to a different
morphology.

### Decide the goal first

There are two different tasks:

- Exact reproduction of a HOC model.
- Reusing the same channel dynamics on a new morphology.

For exact reproduction, copy NEURON's final segment table.

For a new morphology, do not copy old morphology-specific final segment values.
Instead, reapply the intended rules to the new morphology after groups,
distances, and compartments are finalized.

### Build the morphology deliberately

- Preserve section topology if comparing to HOC.
- Avoid artificial cable introduced by SWC conversion.
- Assign section groups explicitly.
- Compute compartment centers after the final compartment count is set.
- Compute path distances from the intended soma root.

### Assign parameters after discretization

For Jaxley-only models, the cleanest rule is:

1. Build morphology.
2. Set final compartment count.
3. Compute distances and groups.
4. Insert channels.
5. Assign passive and active parameters.

This avoids NEURON-style interpolation ambiguity from assigning RANGE variables
before changing `nseg`.

### Audit passive load

Before testing spikes, export and compare:

- number of sections/branches
- number of segments/compartments
- surface area by group
- total `cm * area`
- total `gLeak * area`
- axial resistivity ranges by group
- leak reversal ranges by group

### Audit active placement

For each channel family, compare by group:

- number of compartments with finite channel parameters
- min/max conductance
- area-weighted conductance sum
- distance-dependent parameter ranges
- ion reversal ranges

This catches most placement errors before a full voltage trace is needed.

### Validate in increasing complexity

Recommended current-step checks:

- passive-only, subthreshold
- active, subthreshold, e.g. `0.3 nA`
- active, spike-generating, e.g. `0.9 nA`

At high current, inspect spike timing separately from voltage RMSE. A one-time
step spike offset can produce a large max error even when the models are
otherwise closely aligned.

## Current Alignment Result

After fixing HOC-section morphology, passive load, final active profiles, ion
reversals, and `celsius`, the comparison at `d_lambda = 0.3` produced:

- `0.3 nA`: RMSE about `0.011 mV`, max absolute error about `0.027 mV`.
- `0.9 nA`: RMSE about `0.128 mV`, max absolute error about `6.60 mV`.

The high-current max error occurs around the spike upstroke. The first 0 mV
crossing differed by one `dt = 0.025 ms` step, while peak voltage differed by
about `0.11 mV`.
