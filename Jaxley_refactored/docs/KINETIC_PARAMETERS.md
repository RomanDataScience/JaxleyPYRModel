# Shared kinetic calibration parameters

The Combe calibration catalog contains the original 40 conductance and passive
parameters followed by four positive, dimensionless kinetic time scales. All
four have reference value `1.0`, which reproduces the converted channel
equations without alteration. The default local LSU_1 and hybrid CMA–Adam
pipelines select all 44 parameters.

| Parameter | Bounds | Channel placements | Effect |
| --- | --- | --- | --- |
| `kd_deactivation_tau_scale` | `0.25–4.0` | Kd in soma, apical, axon, basal | Multiplies the fixed 0.6-ms Kd m-gate time constant |
| `nat_fast_inactivation_tau_scale` | `0.5–2.0` | Nav1.6 in soma/apical, Nax in axon, Na3Dend in basal | Slows or accelerates fast sodium inactivation |
| `nat_slow_recovery_tau_scale` | `0.5–2.0` | Nav1.6 in soma/apical | Slows or accelerates entry into and recovery from the deep-inactivated state |
| `h_tau_scale` | `0.5–2.0` | HCN in soma, apical, basal | Multiplies the HCN activation/deactivation time constant |

## Equation semantics

For Kd, the implementation is:

```text
tau_m = 0.6 ms * kd_deactivation_tau_scale
```

The Kd mechanism has one m-gate time constant, so the parameter scales both
activation and deactivation. Its name reflects the DBLO motivation rather than
an isolated deactivation-only gate.

For Nax and Na3Dend, `nat_fast_inactivation_tau_scale` multiplies `tau_h` after
the mechanism's minimum-time-constant floor. Their steady-state `h_inf` is
unchanged. For Markov Nav1.6, the scale divides both directions of the
`O1 <-> I1` and `C1 <-> I1` transition pairs. Pairwise equilibrium ratios are
therefore retained while the fast inactivation transitions change speed.

For Nav1.6 slow kinetics, `nat_slow_recovery_tau_scale` divides both the
`I1 -> I2` and `I2 -> I1` rates. This exactly preserves the equilibrium ratio
between those states, but the parameter changes both slow-inactivation entry
and recovery—not recovery alone. Na3Dend is deliberately excluded because its
slow gate is disabled by the reference model and would provide no calibration
gradient.

For HCN:

```text
tau_hcn = max(reference_tau_hcn, 5 ms) * h_tau_scale
```

The steady-state HCN activation curve is unchanged.

## Interpretation

These are shared phenomenological time-scale parameters. They preserve the
existing regional conductance profiles and avoid fitting dozens of poorly
identifiable rate constants from somatic current clamp alone. Kd deactivation
and transient-sodium kinetics are motivated by the DBLO mechanism study; HCN
kinetics primarily constrain hyperpolarizing sag and return toward baseline.

The ionic reversal potentials and temperature remain fixed. They must not be
used as compensation parameters for DBLO. Parameter estimates should be
interpreted together with sensitivity, boundary-occupancy, and multi-seed
analyses because the available somatic protocols cannot uniquely identify all
44 individual values.

`nap_gnabar` remains a separate fitted conductance parameter. Its reference is
exactly zero and its bounds are `0–0.001 S/cm2`; zero-preserving initialization
does not freeze it during Adam, and CMA–ES can sample positive values.
